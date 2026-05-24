import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import Optional, Dict, Any, Tuple
from tqdm.auto import tqdm

from .configs import PatientZeroConfig

class EpidemicDataAnalyzer:
    
    def __init__(
        self, 
        works_df: pd.DataFrame,
        authors_df: pd.DataFrame,
        citations_df: pd.DataFrame,
        config: Optional[PatientZeroConfig] = None
    ):
        self.works_df = works_df.copy()
        self.citations_df = citations_df.copy()
        self.authors_df = authors_df.copy()
        self.config = config or PatientZeroConfig()
        
        self.works_df.set_index('id', inplace=True)
        self.works_df['referenced_works_list'] = self.works_df['referenced_works'].apply(
            lambda x: x.split(';') if isinstance(x, str) and x else []
        )
        
        self._build_citation_graph()
        
        self.patient_zero_results: Optional[pd.DataFrame] = None
        self.disruption_results: Optional[pd.DataFrame] = None
        self.network_metrics: Optional[pd.DataFrame] = None
        self.semantic_drift_results: Optional[Dict] = None
        
    def _build_citation_graph(self) -> None:

        self.citation_graph = nx.DiGraph()
        
        for work_id in self.works_df.index:
            self.citation_graph.add_node(work_id)
        
        for _, row in self.citations_df.iterrows():
            citing = row['citing_work_id']
            cited = row['cited_work_id']
            if citing in self.works_df.index and cited in self.works_df.index:
                self.citation_graph.add_edge(citing, cited)
        
        logging.info(f"Graph created: {self.citation_graph.number_of_nodes()} nodes, {self.citation_graph.number_of_edges()} edges")
    
    def _min_max_normalize(self, series: pd.Series) -> pd.Series:
        min_val = series.min()
        max_val = series.max()
        if max_val - min_val == 0:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - min_val) / (max_val - min_val)
    
    # === PATIENT ZERO ANALYSIS ===
    
    def identify_patient_zeros(
        self, 
        top_n: int = 10,
        use_disruption: bool = False
    ) -> pd.DataFrame:
        
        if use_disruption:
            self._calculate_persistent_disruption()
            return self._identify_pz_via_disruption(top_n)
        else:
            return self._identify_pz_via_composite(top_n)
    
    def _identify_pz_via_composite(self, top_n: int) -> pd.DataFrame:
        
        infective_mask = self.works_df['state'] == 'infective'
        infective_works = self.works_df[infective_mask].copy()
        
        if len(infective_works) == 0:
            logging.warning("No infective works found")
            return pd.DataFrame()
        
        # Temporal Priority Score
        earliest_year = infective_works['publication_year'].min()
        temporal_score = 1.0 / (infective_works['publication_year'] - earliest_year + 1)
        temporal_score = self._min_max_normalize(temporal_score)
        
        # Backward Dependency
        backward_dep = self._calculate_backward_dependency(infective_works)
        backward_score = 1.0 - self._min_max_normalize(backward_dep)
        
        # Forward Influence
        forward_inf = self._calculate_forward_influence(infective_works)
        forward_score = self._min_max_normalize(forward_inf)
        
        # Citation Score
        citation_score = self._min_max_normalize(
            np.log1p(infective_works['cited_by_count'])
        )
        
        composite_score = (
            self.config.alpha_temporal * temporal_score +
            self.config.alpha_backward * backward_score +
            self.config.alpha_forward * forward_score +
            self.config.alpha_citation * citation_score
        )
        
        self.patient_zero_results = infective_works.copy()[['title', 'abstract', 'publication_year']]
        self.patient_zero_results['temporal_score'] = temporal_score
        self.patient_zero_results['backward_dependency'] = backward_dep
        self.patient_zero_results['forward_influence'] = forward_inf
        self.patient_zero_results['citation_score'] = citation_score
        self.patient_zero_results['composite_pz_score'] = composite_score
        
        top_candidates = self.patient_zero_results.nlargest(top_n, 'composite_pz_score')
        
        return top_candidates
    
    def _calculate_backward_dependency(self, works: pd.DataFrame) -> pd.Series:
        infective_ids = set(works.index)
        backward_deps = []
        
        for work_id in tqdm(works.index, desc="Calculating backward dependency"):
            refs = works.loc[work_id, 'referenced_works_list']
            dep_count = sum(1 for ref in refs if ref in infective_ids and ref != work_id)
            backward_deps.append(dep_count / len(refs) if refs else 0.0)
        
        return pd.Series(backward_deps, index=works.index)
    
    def _calculate_forward_influence(self, works: pd.DataFrame) -> pd.Series:
        infective_ids = set(works.index)
        
        citing_index = {}
        for _, row in self.citations_df.iterrows():
            cited = row['cited_work_id']
            citing = row['citing_work_id']
            if cited not in citing_index:
                citing_index[cited] = []
            citing_index[cited].append(citing)
        
        forward_influence = []
        for work_id in tqdm(works.index, desc="Calculating forward influence"):
            citing_works = citing_index.get(work_id, [])
            if not citing_works:
                forward_influence.append(0.0)
            else:
                infective_citers = sum(1 for cw in citing_works if cw in infective_ids)
                forward_influence.append(infective_citers / len(citing_works))
        
        return pd.Series(forward_influence, index=works.index)

    def _calculate_persistent_disruption(self) -> pd.DataFrame: # можно добавить временное окно, вспомни об этом потом
        
        dr_values = {}
        dc_values = {}
        
        citing_index = {}
        cited_index = {}
        
        for _, row in tqdm(self.citations_df.iterrows(), total=len(self.citations_df), desc="Indexing"):
            citing = row['citing_work_id']
            cited = row['cited_work_id']
            
            if citing not in citing_index:
                citing_index[citing] = []
            citing_index[citing].append(cited)
            
            if cited not in cited_index:
                cited_index[cited] = []
            cited_index[cited].append(citing)
        
        for work_id in tqdm(self.works_df.index, desc="dr calculation"):
            dr_values[work_id] = self._calculate_reference_disruption(
                work_id, citing_index, cited_index
            )
        
        for work_id in tqdm(self.works_df.index, desc="dc calculation"):
            dc_values[work_id] = self._calculate_citation_disruption(
                work_id, citing_index, cited_index
            )
        
        self.disruption_results = pd.DataFrame({
            'dr': dr_values,
            'dc': dc_values
        })
        self.disruption_results['disruption_type'] = self.disruption_results.apply(
            self._classify_disruption_type, axis=1
        )
        
        self.disruption_results = self.disruption_results.join(
            self.works_df[['publication_year', 'cited_by_count', 'state']]
        )
        
        return self.disruption_results
    
    def _calculate_reference_disruption(
        self, 
        work_id: str, 
        citing_index: Dict, 
        cited_index: Dict
    ) -> float:

        refs = self.works_df.loc[work_id, 'referenced_works_list']
        if not refs:
            return 0.0
        
        work_year = self.works_df.loc[work_id, 'publication_year']
        if pd.isna(work_year):
            return 0.0
        
        d_values = []
        
        for ref_id in refs:
            if ref_id not in self.works_df.index:
                continue
            
            ref_year = self.works_df.loc[ref_id, 'publication_year']
            if pd.isna(ref_year) or ref_year >= work_year:
                continue
            
            work_citers = set(cited_index.get(work_id, []))
            ref_citers = set(cited_index.get(ref_id, []))
            
            dc_ij = len(work_citers - ref_citers)
            
            cc_ij = len(work_citers & ref_citers)
            
            rc_ij = 0
            for citer in ref_citers:
                citer_year = self.works_df.loc[citer, 'publication_year']
                if not pd.isna(citer_year) and citer_year > work_year:
                    if citer not in work_citers:
                        rc_ij += 1
            
            denominator = dc_ij + cc_ij + rc_ij
            if denominator > 0:
                d_ij = (dc_ij - cc_ij) / denominator
                d_values.append(d_ij)
        
        return np.mean(d_values) if d_values else 0.0
    
    def _calculate_citation_disruption(
        self, 
        work_id: str, 
        citing_index: Dict, 
        cited_index: Dict
    ) -> float:
        
        citers = cited_index.get(work_id, [])
        if not citers:
            return 0.0
        
        work_year = self.works_df.loc[work_id, 'publication_year']
        if pd.isna(work_year):
            return 0.0
        
        d_values = []
        
        for citer_id in citers:
            if citer_id not in self.works_df.index:
                continue
            
            citer_year = self.works_df.loc[citer_id, 'publication_year']
            if pd.isna(citer_year) or citer_year <= work_year:
                continue
            
            citer_citers = set(cited_index.get(citer_id, []))
            work_citers = set(cited_index.get(work_id, []))
            
            dc_ki = len(citer_citers - work_citers)
            
            cc_ki = len(citer_citers & work_citers)
            
            rc_ki = 0
            for citer2 in work_citers:
                citer2_year = self.works_df.loc[citer2, 'publication_year']
                if not pd.isna(citer2_year) and citer2_year > citer_year:
                    if citer2 not in citer_citers:
                        rc_ki += 1
            
            denominator = dc_ki + cc_ki + rc_ki
            if denominator > 0:
                d_ki = (dc_ki - cc_ki) / denominator
                d_values.append(d_ki)
        
        return np.mean(d_values) if d_values else 0.0
    
    def _classify_disruption_type(self, row: pd.Series) -> str:

        dr_q75 = self.disruption_results['dr'].quantile(0.75)
        dr_q25 = self.disruption_results['dr'].quantile(0.25)
        dc_q25 = self.disruption_results['dc'].quantile(0.25)
        dc_q75 = self.disruption_results['dc'].quantile(0.75)
        
        dr = row['dr']
        dc = row['dc']
        
        if dr >= dr_q75 and dc <= dc_q25:
            return "persistently_disruptive"
        elif dr >= dr_q75 and dc >= dc_q75:
            return "disrupted"
        elif dr <= dr_q25:
            return "developmental"
        else:
            return "moderate"
    
    def _identify_pz_via_disruption(self, top_n: int) -> pd.DataFrame:

        mask = (
            (self.disruption_results['dr'] >= self.disruption_results['dr'].quantile(0.75)) &
            (self.disruption_results['dc'] <= self.disruption_results['dc'].quantile(0.25))
        )
        
        pd_works = self.disruption_results[mask].copy()
        
        pd_works['pd_score'] = (
            self._min_max_normalize(pd_works['dr']) * 0.6 +
            self._min_max_normalize(np.log1p(pd_works['cited_by_count'])) * 0.4
        )
        
        top_candidates = pd_works.nlargest(top_n, 'pd_score')
        
        return top_candidates
    
    
    # === NETWORK METRICS ===
    
    def compute_network_metrics(self) -> pd.DataFrame:
        
        metrics = {}
        
        in_degree = dict(self.citation_graph.in_degree())
        out_degree = dict(self.citation_graph.out_degree())
        metrics['in_degree'] = pd.Series(in_degree)
        metrics['out_degree'] = pd.Series(out_degree)
        
        # PageRank
        pagerank = nx.pagerank(self.citation_graph, alpha=0.85)
        metrics['pagerank'] = pd.Series(pagerank)
        
        # Clustering coefficient
        undirected_graph = self.citation_graph.to_undirected()
        clustering = nx.clustering(undirected_graph)
        metrics['clustering_coef'] = pd.Series(clustering)
        
        # Betweenness centrality
        betweenness = nx.betweenness_centrality(self.citation_graph)
        metrics['betweenness'] = pd.Series(betweenness)
        
        self.network_metrics = pd.DataFrame(metrics)
        
        return self.network_metrics
    
    # === SEMANTIC DRIFT ===
    
    def analyze_semantic_drift(self) -> Dict:
        
        infective_works = self.works_df[self.works_df['state'] == 'infective'].copy()
        
        if len(infective_works) == 0:
            logging.warning("No infective works found")
            return {}
        
        
        correlation = infective_works['similarity'].corr(infective_works['depth'])
        depth_stats = infective_works.groupby('depth')['similarity'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        depth_0_sim = infective_works[infective_works['depth'] == 0]['similarity'].mean()
        depth_max = infective_works['depth'].max()
        depth_max_sim = infective_works[infective_works['depth'] == depth_max]['similarity'].mean()
        
        drift_magnitude = depth_0_sim - depth_max_sim
        
        depth_0_group = infective_works[infective_works['depth'] == 0]['similarity']
        depth_max_group = infective_works[infective_works['depth'] == depth_max]['similarity']
        
        if len(depth_0_group) > 1 and len(depth_max_group) > 1:
            t_stat, p_value = stats.ttest_ind(depth_0_group, depth_max_group)
        else:
            t_stat, p_value = np.nan, np.nan
        
        self.semantic_drift_results = {
            'correlation_depth_similarity': correlation,
            'depth_statistics': depth_stats,
            'drift_magnitude': drift_magnitude,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_drift': p_value < 0.05 if not np.isnan(p_value) else False
        }
        
        return self.semantic_drift_results

    # === COMPARATIVE ANALYSIS ===
    
    def comparative_analysis(self, group_by: str = 'state') -> pd.DataFrame:
        
        df = self.works_df.copy()
        
        if self.network_metrics is not None:
            network_cols = ['in_degree', 'pagerank', 'clustering_coef', 'betweenness']
            df = df.join(self.network_metrics[network_cols])
        
        if self.disruption_results is not None:
            df = df.join(self.disruption_results[['dr', 'dc', 'disruption_type']])
        
        if group_by == 'state':
            grouped = df.groupby('state').agg({
                'cited_by_count': ['mean', 'median', 'std'],
                'similarity': ['mean', 'std'],
                'publication_year': ['min', 'max', 'count']
            })
        
        elif group_by == 'depth':
            grouped = df.groupby('depth').agg({
                'cited_by_count': 'mean',
                'similarity': 'mean',
                'id': 'count'
            }).rename(columns={'id': 'count'})
        
        return grouped

    def population_segmentation(
        self,
        field_col: str = 'main_topic',
        min_works: int = 10,
        include_temporal: bool = True
    ) -> Dict[str, Any]:
        
        if field_col not in self.authors_df.columns:
            logging.warning(f"Column '{field_col}' not found in authors_df. "
                           f"Available: {list(self.authors_df.columns)}")
            return {}
        
        field_groups = self.authors_df.groupby(field_col)
        
        field_stats = []
        field_temporal = {}
        field_network_metrics = {}
        
        for field, group in field_groups:
            if field is None or pd.isna(field) or str(field).strip() == '':
                continue
            
            num_authors = len(group)
            if num_authors < min_works:
                continue
            
            num_infective = group['is_infective'].sum() if 'is_infective' in group.columns else 0
            num_susceptible = group['is_susceptible'].sum() if 'is_susceptible' in group.columns else 0
            avg_works = group['total_works'].mean() if 'total_works' in group.columns else 0

            first_year = group['first_publication_year'].min() if 'first_publication_year' in group.columns else None
            last_year = group['last_publication_year'].max() if 'last_publication_year' in group.columns else None

            infective_ratio = num_infective / num_authors if num_authors > 0 else 0
            
            field_stats.append({
                'field': field,
                'num_authors': num_authors,
                'num_infective': int(num_infective),
                'num_susceptible': int(num_susceptible),
                'avg_works_per_author': round(avg_works, 2),
                'infective_ratio': round(infective_ratio, 3),
                'first_year': int(first_year) if pd.notna(first_year) else None,
                'last_year': int(last_year) if pd.notna(last_year) else None,
                'time_span': (int(last_year) - int(first_year) + 1) 
                            if pd.notna(first_year) and pd.notna(last_year) else None
            })
            
            if include_temporal and 'first_infective_year' in group.columns:
                infective_authors = group[group['is_infective'] == True]
                if len(infective_authors) > 0:
                    temporal = infective_authors.groupby('first_infective_year').size()
                    field_temporal[field] = temporal
            
            if self.network_metrics is not None:
                field_works = self.works_df[
                    self.works_df['primary_topic'] == field
                ].index.tolist()
                if field_works:
                    subgraph = self.citation_graph.subgraph(field_works)
                    field_network_metrics[field] = {
                        'density': nx.density(subgraph),
                        'avg_clustering': nx.average_clustering(subgraph.to_undirected())
                    }
        
        field_stats_df = pd.DataFrame(field_stats)
        if field_stats_df.empty:
            logging.warning(f"No fields found with >= {min_works} authors")
            return {}
        
        field_rankings = {
            'by_infective_count': field_stats_df.nlargest(10, 'num_infective')['field'].tolist(),
            'by_infective_ratio': field_stats_df.nlargest(10, 'infective_ratio')['field'].tolist(),
            'by_author_count': field_stats_df.nlargest(10, 'num_authors')['field'].tolist(),
            'by_avg_works': field_stats_df.nlargest(10, 'avg_works_per_author')['field'].tolist()
        }
        
        results = {
            'field_stats': field_stats_df,
            'field_temporal': field_temporal,
            'field_network_metrics': field_network_metrics,
            'field_rankings': field_rankings,
            'num_fields': len(field_stats_df),
            'total_authors_analyzed': field_stats_df['num_authors'].sum()
        }
        
        return results
    
    # === VISUALIZATION ===
    
    def plot_patient_zero_timeline(
        self, 
        top_n: int = 10, 
        show: bool = True,
        mode: str = 'composite_pz_score'
    ) -> plt.Figure:
        """
        Визуализация временной шкалы Patient Zero.
        """
        if self.patient_zero_results is None:
            logging.warning("Run identify_patient_zeros() first")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        top_candidates = self.patient_zero_results.nlargest(top_n, mode)
        
        years = top_candidates['publication_year']
        scores = top_candidates['composite_pz_score']
        citations = np.log1p(top_candidates['cited_by_count'])
        
        scatter = ax.scatter(
            years, scores, 
            s=citations * 50, 
            c=citations, 
            cmap='Reds', 
            alpha=0.6,
            edgecolors='black',
            linewidth=1
        )
        
        ax.set_xlabel('publication year', fontsize=12)
        ax.set_ylabel(mode, fontsize=12)
        ax.set_title(f'Top {top_n} Patient Zero candidates', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Log Citations', fontsize=10)
        
        plt.tight_layout()
        if show:
            plt.show()
        
        return fig
    
    def plot_disruption_space(self, show: bool = True) -> plt.Figure:
        
        if self.disruption_results is None:
            logging.warning("Run calculate_persistent_disruption() first")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        type_colors = {
            'persistently_disruptive': 'red',
            'disrupted': 'orange',
            'developmental': 'blue',
            'moderate': 'gray'
        }
        
        for dtype in self.disruption_results['disruption_type'].unique():
            mask = self.disruption_results['disruption_type'] == dtype
            subset = self.disruption_results[mask]
            
            ax.scatter(
                subset['dr'], subset['dc'],
                c=type_colors.get(dtype, 'gray'),
                label=dtype,
                alpha=0.5,
                s=30,
                edgecolors='none'
            )
        
        dr_q75 = self.disruption_results['dr'].quantile(0.75)
        dc_q25 = self.disruption_results['dc'].quantile(0.25)
        
        ax.axvline(x=dr_q75, color='green', linestyle='--', linewidth=2, label='Thresholds')
        ax.axhline(y=dc_q25, color='green', linestyle='--', linewidth=2)
        
        ax.axvspan(dr_q75, 1, alpha=0.1, color='red')
        ax.axhspan(0, dc_q25, alpha=0.1, color='red')
        
        ax.set_xlabel('reference disruption (dr)', fontsize=12)
        ax.set_ylabel('citation disruption (dc)', fontsize=12)
        ax.set_title('persistent disruption space', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    def plot_epidemic_curve(self, show: bool = True) -> plt.Figure:

        fig, ax = plt.subplots(figsize=(12, 6))
        
        yearly_counts = self.works_df.groupby(['publication_year', 'state']).size().unstack(fill_value=0)
        
        if 'infective' in yearly_counts.columns:
            ax.plot(
                yearly_counts.index, 
                yearly_counts['infective'], 
                'o-', 
                linewidth=2, 
                markersize=6,
                label='Infective',
                color='red'
            )
        
        if 'susceptible' in yearly_counts.columns:
            ax.plot(
                yearly_counts.index, 
                yearly_counts['susceptible'], 
                's-', 
                linewidth=2, 
                markersize=6,
                label='Susceptible',
                color='blue'
            )
        
        ax.set_xlabel('publication year', fontsize=12)
        ax.set_ylabel('number of works', fontsize=12)
        ax.set_title('Epidemic curve', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    def plot_network_metrics_distribution(self, show: bool = True) -> plt.Figure:

        if self.network_metrics is None:
            logging.warning("Run compute_network_metrics() first")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics_to_plot = ['in_degree', 'pagerank', 'clustering_coef', 'betweenness']
        titles = ['In-degree distribution', 'PageRank distribution', 
                 'Clustering coefficient', 'Betweenness centrality']
        
        for ax, metric, title in zip(axes.flat, metrics_to_plot, titles):
            ax.hist(
                self.network_metrics[metric].dropna(), 
                bins=50, 
                alpha=0.7, 
                edgecolor='black'
            )
            ax.set_xlabel(metric, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig

    def plot_field_comparison(
        self,
        segmentation_results: Dict[str, Any],
        top_n: int = 15,
        show: bool = True
    ) -> Tuple[plt.Figure, plt.Figure]:
        
        if not segmentation_results or 'field_stats' not in segmentation_results:
            logging.warning("No segmentation results to plot")
            return None, None
        
        df = segmentation_results['field_stats'].copy()
        
        # === Infective authors ===
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        top_fields = df.nlargest(top_n, 'num_infective')
        
        bars1 = ax1.barh(
            top_fields['field'], 
            top_fields['num_infective'],
            color='steelblue',
            edgecolor='navy',
            alpha=0.8
        )
        ax1.set_xlabel('Number of infective authors', fontsize=11)
        ax1.set_title(f'Top {top_n} fields by infective authors', fontsize=13, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        for bar in bars1:
            width = bar.get_width()
            ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', va='center', fontsize=9)
        
        scatter = ax2.scatter(
            df['num_authors'], 
            df['infective_ratio'],
            s=df['num_infective'] * 10,
            c=df['infective_ratio'],
            cmap='viridis',
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
        ax2.set_xlabel('Total authors in field', fontsize=11)
        ax2.set_ylabel('Infective ratio', fontsize=11)
        ax2.set_title('Infective Ratio vs Size', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        median_ratio = df['infective_ratio'].median()
        ax2.axhline(y=median_ratio, color='red', linestyle='--', 
                   label=f'Median: {median_ratio:.1%}', linewidth=1.5)
        ax2.legend(fontsize=9)
        
        plt.tight_layout()
        
        # === Temporal ===
        fig2 = None
        if segmentation_results.get('field_temporal'):
            fig2, ax = plt.subplots(figsize=(12, 6))
            
            temporal_data = segmentation_results['field_temporal']
            colors = plt.cm.Set2(np.linspace(0, 1, min(len(temporal_data), top_n)))
            
            top_field_names = df.nlargest(top_n, 'num_infective')['field'].tolist()
            
            for idx, (field, series) in enumerate(temporal_data.items()):
                if field not in top_field_names:
                    continue
                ax.plot(
                    series.index, series.values,
                    label=field,
                    color=colors[idx % len(colors)],
                    linewidth=2,
                    marker='o' if len(series) < 15 else None,
                    markersize=4
                )
            
            ax.set_xlabel('Year of first infective publication', fontsize=11)
            ax.set_ylabel('New infective authors', fontsize=11)
            ax.set_title('Temporal dynamics', 
                        fontsize=13, fontweight='bold')
            ax.legend(loc='upper left', fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            
            plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig1, fig2

    def plot_field_network_properties(
        self,
        segmentation_results: Dict[str, Any],
        show: bool = True
    ) -> Optional[plt.Figure]:
    
        if not segmentation_results.get('field_network_metrics'):
            logging.warning("No network metrics available for fields")
            return None
        
        metrics = segmentation_results['field_network_metrics']
        
        if not metrics:
            return None
        
        plot_data = []
        for field, props in metrics.items():
            plot_data.append({
                'field': field,
                'density': props.get('density', 0),
                'clustering': props.get('avg_clustering', 0)
            })
        
        df = pd.DataFrame(plot_data)
        if len(df) < 3:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(
            df['density'], 
            df['clustering'],
            s=100,
            c=df['density'],
            cmap='plasma',
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5
        )
        
        for _, row in df.iterrows():
            ax.annotate(
                row['field'][:20] + ('...' if len(row['field']) > 20 else ''),
                (row['density'], row['clustering']),
                fontsize=8,
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
        
        ax.set_xlabel('Network density', fontsize=11)
        ax.set_ylabel('Average clustering coefficient', fontsize=11)
        ax.set_title('Network structure by scientific field', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax.axvline(x=df['density'].median(), color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=df['clustering'].median(), color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
