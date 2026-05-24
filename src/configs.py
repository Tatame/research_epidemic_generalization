import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class SeedScorerConfig:
    alpha_sem: float = 0.8
    alpha_cit: float = 0.15
    alpha_meta: float = 0.05
    seed_percentile: float = 0.50
    expansion_infected_percentile: float = 0.75
    expansion_susceptible_percentile: float = 0.45
    batch_size: int = 16
    
    def __post_init__(self):
        if not np.isclose(self.alpha_sem + self.alpha_cit + self.alpha_meta, 1.0):
            raise ValueError("Scoring weights must sum to 1.0")
        if not (0.0 <= self.seed_percentile <= 0.99):
            raise ValueError("seed_percentile must be in [0.2, 0.99]")

@dataclass
class ExpansionConfig:
    tau_infective: float = 0.85
    tau_susceptible: float = 0.6
    max_depth: int = 2
    max_works: int = 50_000
    max_refs_per_work: int = 15
    max_cit_per_work: int = 15
    
    def __post_init__(self):
        if self.tau_susceptible >= self.tau_infective:
            raise ValueError("susceptible_threshold must be < min_similarity")
        if self.max_refs_per_work < 1:
            raise ValueError("max_refs_per_work must be >= 1")

@dataclass
class PatientZeroConfig:
    alpha_temporal: float = 0.1
    alpha_backward: float = 0.30
    alpha_forward: float = 0.40
    alpha_citation: float = 0.20
    
    early_window_years: int = 2
    
    dr_percentile: float = 0.75
    dc_percentile: float = 0.25
    
    def __post_init__(self):
        if not np.isclose(self.alpha_temporal + self.alpha_backward + 
                         self.alpha_forward + self.alpha_citation, 1.0):
            raise ValueError("Weights must sum to 1.0")
