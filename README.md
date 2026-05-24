# Epidemic Generalization of Scientific Knowledge Transmission

## Overview
This repository implements a computational framework for automated extraction, structural analysis, and dynamic modeling of scientific domains. The approach generalizes the epidemic theory of idea transmission (Goffman & Newill, 1964) by combining semantic-citation graph traversal, persistent disruption analysis (Deng et al., 2025), and Bayesian parameter estimation of coupled host-vector differential equations.

The system operates without manual curation: a natural language query defines the target idea, a hybrid scoring mechanism selects seed publications, and a bidirectional BFS expands the domain through citation networks while preserving semantic coherence. Subsequent modules identify "patient zero" works, compute topological and disruption metrics, and forecast publication dynamics using a probabilistic ODE solver.

## Project Structure

`src`/  
├── `init.py` # Public API exports  
├── `models.py` # Data schemas (WorkMetadata)  
├── `configs.py` # Immutable configuration classes  
├── `collector.py` # PyAlex API, hybrid scoring, semantic BFS, CSV export  
├── `analyzer.py` # Patient zero identification, disruption metrics, network analysis, visualization  
└── `ode_model.py` # Time-series preparation, Bayesian ODE parameter estimation  

_Note: sentence-transformers and pymc require GPU access for optimal performance. Set `device = "cuda"` in `collector.py` if available._

## Usage

### 1. Domain Collection

1. **Query → Candidate Pool**: PyAlex retrieves articles matching semantic filters.
2. **Hybrid Scoring → Seeds**: Cosine similarity + citation log-scale + token overlap determine initial infectives.
3. **BFS Expansion → Graph**: Bidirectional traversal classifies works as infective/susceptible based on adaptive thresholds.
4. **Aggregation → CSV**: Works, authors, and citation edges are exported for analysis.

```python
from src import EpidemicOfScientificKnowledgeModel, SeedScorerConfig, ExpansionConfig
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("allenai/specter2_base")
collector = EpidemicOfScientificKnowledgeModel(model=model, bidirectional=True)

tau_i, tau_s = collector.get_seed_works_by_query(
    query="YUOR QUERY",
    config=SeedScorerConfig(...),
    year_range=...,
    max_candidates_by_query=...
)

collector.semantic_expansion_bfs(config=ExpansionConfig(...))
collector.save_results(field_name="YOUR FIELD NAME")
```

**Configuration parameters**:  

`SeedScorerConfig`: Controls hybrid scoring for seed selection. Weights must sum to 1.0. _Higher `alpha_sem` prioritizes semantic relevance; higher `alpha_cit` favors established works; higher `alpha_meta` ensures keyword alignment._  
`ExpansionConfig`: Controls BFS expansion. _`tau_infective` and `tau_susceptible` define default semantic boundaries; `max_depth` limits citation distance from seeds (prevents semantic drift)._

### 2. Structural Analysis

```python
import pandas as pd
from src import EpidemicDataAnalyzer, PatientZeroConfig

works = pd.read_csv("DOMAIN_NAME_works.csv")
authors = pd.read_csv("DOMAIN_NAME_authors.csv")
citations = pd.read_csv("DOMAIN_NAME_citations.csv")

analyzer = EpidemicDataAnalyzer(works, authors, citations, PatientZeroConfig(...))
patient_zeros = analyzer.identify_patient_zeros(top_n=...)
analyzer.compute_network_metrics()
...
```

**Configuration parameters**:  

`PatientZeroConfig`: Controls composite scoring for identifying foundational works. Weights must sum to 1.0.  
`alpha_temporal`: Prioritizes early works in the domain  
`alpha_backward`: Rewards works independent of prior literature (low backward citations)  
`alpha_forward : Rewards works that influenced many subsequent infective works  
`alpha_citation : Rewards high citation impact  
`dr_percentile`/`dc_percentile`: Define thresholds for persistent disruption quadrant (high dr, low dc)  

### 3. Epidemic model

```python
from src import prepare_yearly_timeseries, run_validation_experiment, run_forecast_with_interpretation

df_data = prepare_yearly_timeseries(works, start_year=..., end_year=...)

empirical_priors = calculate_empirical_priors(df_data)

model, trace, t_full, n_obs, y0, mu, mu_p = build_and_sample_model(
    df_data, empirical_priors=empirical_priors
)
```

## Reproducibility

* API rate limits and caching are embedded in `collector.py`.
* Configuration dataclasses enforce weight normalization and threshold bounds at initialization.
* Results are logged with timestamps and parameter states.
