# Pipeline API Contract

## Module: `src.models.vae`
```python
def build_vae_model(latent_dim: int = 16, num_categories: int, num_temporal_features: int) -> torch.nn.Module:
    """Constructs the feedforward Variational Autoencoder."""
    pass
```

## Module: `src.services.baseline`
```python
def train_baseline_vae(data: pd.DataFrame, model: torch.nn.Module, epochs: int) -> torch.nn.Module:
    """Trains the VAE on historical purchase data without stimuli. Implements explicit Reparameterization Trick and KL + Recon loss."""
    pass

def get_household_profile(model: torch.nn.Module, household_data: pd.DataFrame) -> np.ndarray:
    """Returns the latent representation (mu vector) for a given household."""
    pass
```

## Module: `src.services.impact_analysis`
```python
def calculate_deviation(baseline_profile: np.ndarray, post_stimulus_data: pd.DataFrame, model: torch.nn.Module) -> float:
    """Calculates the quantitative magnitude of shift (e.g., distance in latent space)."""
    pass

def categorize_shift(baseline_data: pd.DataFrame, post_stimulus_data: pd.DataFrame) -> str:
    """Categorizes the qualitative nature of shift into predefined categories based on rules."""
    pass

def analyze_persistence(household_data: pd.DataFrame, stimulus_end: pd.Timestamp, model: torch.nn.Module, baseline_profile: np.ndarray) -> int:
    """Returns the number of days until the household behavior returns to the baseline."""
    pass
```

## Module: `src.services.reporting`
```python
def segment_consumers(shifts: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """Groups consumers into segments based on similarity in their reaction patterns using clustering algorithms (e.g., K-Means)."""
    pass

def rank_sensitive_categories(shifts: pd.DataFrame, transactions: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Identifies and ranks the top product categories based on their sensitivity (degree of deviation) to external shifts."""
    pass
```