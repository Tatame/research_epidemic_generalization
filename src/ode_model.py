import pymc as pm
import pytensor.tensor as pt
from pytensor.graph.op import Op
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import arviz as az
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def prepare_yearly_timeseries(works_df: pd.DataFrame, authors_df: pd.DataFrame = None, 
                              start_year: int = None, end_year: int = None) -> pd.DataFrame:

    works = works_df.copy()    
    works = works[works['publication_year'].notna()]
    works['publication_year'] = works['publication_year'].astype(int)
    if start_year: works = works[works['publication_year'] >= start_year]
    if end_year: works = works[works['publication_year'] <= end_year]
    if works.empty:
        raise ValueError("There are no data after filtration")
    
    # works
    vec_counts = works.groupby(['publication_year', 'broad']).size().unstack(fill_value=0)
    full_years = range(vec_counts.index.min(), vec_counts.index.max() + 1)
    vec_counts = vec_counts.reindex(full_years, fill_value=0)
    S_prime = vec_counts.get(1, pd.Series(0, index=vec_counts.index)).astype(float)
    I_prime = vec_counts.get(0, pd.Series(0, index=vec_counts.index)).astype(float)
    
    # authors
    author_states = works[['publication_year', 'authors', 'broad']].copy()
    author_states['authors'] = author_states['authors'].astype(str).str.split(';')
    author_states = author_states.explode('authors')
    author_states = author_states[author_states['authors'].notna() & (author_states['authors'].str.strip() != '')]
    author_year_state = author_states.sort_values('broad').groupby(['publication_year', 'authors'])['broad'].first()
    author_year_state = author_year_state.reset_index()
    host_counts = author_year_state.groupby(['publication_year', 'broad']).size().unstack(fill_value=0)
    host_counts = host_counts.reindex(full_years, fill_value=0)
    S = host_counts.get(1, pd.Series(0, index=host_counts.index)).astype(float)
    I = host_counts.get(0, pd.Series(0, index=host_counts.index)).astype(float)
    
    df = pd.DataFrame({
        'year': list(full_years), 'S': S.values, 'I': I.values,
        'S_prime': S_prime.values, 'I_prime': I_prime.values
    })
    return df[df['year'] > 0].reset_index(drop=True)


# === ODE ===
def ode_system(t, y, beta, gamma, beta_p, gamma_p, mu, mu_p):
    S, I, R, Sp, Ip, Rp = y
    
    # authors
    dS  = -beta * S * Ip + mu
    dI  =  beta * S * Ip - gamma * I
    dR  =  gamma * I
    
    # works
    dSp = -beta_p * Sp * I + mu_p
    dIp =  beta_p * Sp * I - gamma_p * Ip
    dRp =  gamma_p * Ip
    
    return [dS, dI, dR, dSp, dIp, dRp]


class ODESolverOp(Op):
    itypes = [pt.dvector]  # [beta, gamma, beta_p, gamma_p]
    otypes = [pt.dmatrix]  # (n_timesteps, 6_states)
    
    def __init__(self, t_eval, y0, mu, mu_p):
        self.t_eval = np.asarray(t_eval, dtype=np.float64)
        self.y0 = np.asarray(y0, dtype=np.float64)
        self.mu = mu
        self.mu_p = mu_p
    
    def perform(self, node, inputs, outputs):
        params = inputs[0]
        beta, gamma, beta_p, gamma_p = params
        try:
            sol = solve_ivp(
                lambda t, y: ode_system(t, y, beta, gamma, beta_p, gamma_p, self.mu, self.mu_p),
                t_span=(self.t_eval[0], self.t_eval[-1]),
                y0=self.y0, t_eval=self.t_eval, method='RK45',
                rtol=1e-4, atol=1e-6, max_step=1.0
            )
            outputs[0][0] = sol.y.T
        except Exception:
            outputs[0][0] = np.zeros((len(self.t_eval), 6))

# === MODEL ===

def calculate_mu(df: pd.DataFrame) -> tuple[float, float]:
    total_authors = df['S'] + df['I']
    total_papers = df['S_prime'] + df['I_prime']
    mu = float(np.mean(np.diff(total_authors.values)))
    mu_p = float(np.mean(np.diff(total_papers.values)))
    return max(0, mu), max(0, mu_p)


def calculate_empirical_priors(df: pd.DataFrame) -> dict:

    df_calc = df[df['I'] > 1e-6].copy()
    if len(df_calc) < 3:
        print("!!!0!!!")
        return None
    
    dI = np.diff(df_calc['I'].values)
    I_lag = df_calc['I'].values[:-1]
    S_lag = df_calc['S'].values[:-1]
    Ip_lag = df_calc['I_prime'].values[:-1]
    
    y = dI / I_lag
    x = (S_lag * Ip_lag) / I_lag
    
    try:
        coeffs, _, _, _ = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)
        beta_est = max(coeffs[0], 1e-6)
        gamma_est = max(-coeffs[1], 0.01)
    except:
        print("!!!1!!!")
        return None
    
    df_calc_p = df[df['I_prime'] > 1e-6].copy()
    if len(df_calc_p) < 3:
        print("!!!2!!!")
        return None
        
    dIp = np.diff(df_calc_p['I_prime'].values)
    Ip_lag = df_calc_p['I_prime'].values[:-1]
    S_p_lag = df_calc_p['S_prime'].values[:-1]
    I_lag_p = df_calc_p['I'].values[:-1]
        
    y_p = dIp / Ip_lag
    x_p = (S_p_lag * I_lag_p) / Ip_lag
        
    try:
        coeffs_p, _, _, _ = np.linalg.lstsq(np.vstack([x_p, np.ones_like(x_p)]).T, y_p, rcond=None)
        beta_p_est = max(coeffs_p[0], 1e-7)
        gamma_p_est = max(-coeffs_p[1], 0.01)
    except:
        print("!!!3!!!")
        return None
        
    priors = {
        'beta':   {'scale': max(beta_est * 10, 1e-4)},
        'gamma':  {'scale': max(gamma_est * 5, 0.1)},
        'beta_p': {'scale': max(beta_p_est * 10, 1e-5)},
        'gamma_p':{'scale': max(gamma_p_est * 5, 0.1)},
    }
    return priors

    
def build_and_sample_model(df: pd.DataFrame, forecast_years: int = 1, 
                           draws: int = 2000, tune: int = 1000,
                           empirical_priors: dict = None) -> tuple:

    t_train = df['year'].values.astype(float)
    n_obs = len(t_train)
    t_forecast = np.arange(t_train[-1] + 1, t_train[-1] + 1 + forecast_years, dtype=float)
    t_full = np.concatenate([t_train, t_forecast])
    
    y0 = np.array([
        df.iloc[0]['S'], df.iloc[0]['I'], 0.0,
        df.iloc[0]['S_prime'], df.iloc[0]['I_prime'], 0.0
    ])
    
    mu_val, mu_p_val = calculate_mu(df)
    solver = ODESolverOp(t_full, y0, mu_val, mu_p_val)
    
    with pm.Model() as model:

        if empirical_priors is None:
            beta   = pm.HalfNormal('beta', sigma=1e-2)
            gamma  = pm.HalfNormal('gamma', sigma=0.5)
            beta_p = pm.HalfNormal('beta_p', sigma=1e-3)
            gamma_p= pm.HalfNormal('gamma_p', sigma=0.5)
        else:
            beta   = pm.HalfNormal('beta', sigma=empirical_priors['beta']['scale'])
            gamma  = pm.HalfNormal('gamma', sigma=empirical_priors['gamma']['scale'])
            beta_p = pm.HalfNormal('beta_p', sigma=empirical_priors['beta_p']['scale'])
            gamma_p= pm.HalfNormal('gamma_p', sigma=empirical_priors['gamma_p']['scale'])
        
        sigma = pm.HalfNormal('sigma', sigma=df['I'].std() * 0.5)
        
        params = pm.math.stack([beta, gamma, beta_p, gamma_p])
        ode_sol = solver(params)
        
        I_pred   = ode_sol[:, 1]
        I_p_pred = ode_sol[:, 4]
        
        pm.Normal('obs_I', mu=I_pred[:n_obs], sigma=sigma, observed=df['I'].values)
        pm.Normal('obs_Ip', mu=I_p_pred[:n_obs], sigma=sigma, observed=df['I_prime'].values)
        
        trace = pm.sample(
            draws=draws, tune=tune, chains=4, target_accept=0.85,
            cores=1, init='adapt_diag', random_seed=42, progressbar=True
        )
    
    return model, trace, t_full, n_obs, y0, mu_val, mu_p_val
