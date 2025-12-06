# Density Estimation Implementation Plan for DeepHall_disorder

## 1. Analysis

### 1.1 Architectural Differences & Symmetry Breaking
The core difference between `DeepHall` (clean) and `DeepHall_disorder` is the presence of fixed charged impurities.
- **Clean System**: The Hamiltonian is rotationally invariant. The ground state electron density $\rho(\mathbf{r})$ is typically uniform (constant) or, if symmetry is broken spontaneously (e.g., stripes), it is often averaged or effectively only dependent on $\theta$ after integration. The existing density estimator histograms only the polar angle $\theta$, relying on the symmetry $\rho(\mathbf{r}) = \rho(\theta)$ (or integrating out $\phi$).
- **Disordered System**: The impurity configuration breaks continuous rotational symmetry. The disorder potential $V_{\text{dis}}(\mathbf{r})$ creates localized potential valleys and hills. Consequently, the electron density $\rho(\mathbf{r})$ is a function of both $\theta$ and $\phi$: $\rho(\mathbf{r}) = \rho(\theta, \phi)$.

### 1.2 Mathematical Implications
The existing approach of calculating $\rho(\theta)$ via 1D histogramming is invalid because:
1.  It integrates out the azimuthal angle $\phi$, washing out the specific localization features we want to observe (electrons pinning to impurities).
2.  The density is not uniform in $\phi$.

We must calculate the 2D density $\rho(\theta, \phi)$ on the sphere.
- **Probability Density**: MCMC samples distribute according to $|\Psi|^2$.
- **Histogram**: We collect 2D bin counts $N_{ij}$ for bins defined by intervals $[\theta_i, \theta_{i+1}]$ and $[\phi_j, \phi_{j+1}]$.
- **Physical Density**:
  $$ \rho(\theta, \phi) \propto \frac{N_{ij}}{d\Omega_{ij}} = \frac{N_{ij}}{\Delta\phi (\cos\theta_i - \cos\theta_{i+1})} $$
  or approximately $N_{ij} / (\sin\theta_i \Delta\theta \Delta\phi)$.

### 1.3 Data Requirements
To reconstruct the density, we need:
1.  **Observable**: A new MCMC observable that histograms electron positions in 2D $(\theta, \phi)$.
2.  **Disorder Config**: The positions of impurities to correlate electron density peaks with impurity locations. This is already saved in `disorder_config.npz`.

---

## 2. Implementation Plan

### 2.1 Task 1: Create `Density2DEstimator`
**File**: `deephall/netobs_bridge/observables/density2d.py`

We will implement a new estimator class `Density2DEstimator` that inherits from `Estimator`.

**Key Features**:
- **Bins**: Configurable tuple `(n_theta, n_phi)`, defaulting to e.g., `(50, 100)`.
- **Range**: $\theta \in [0, \pi]$, $\phi \in [-\pi, \pi]$ (or $[0, 2\pi]$ depending on convention). DeepHall uses `arctan2` for phi usually, so $[-\pi, \pi]$ is standard.
- **Logic**:
  - Extract $\theta = \text{data}[..., 0]$ and $\phi = \text{data}[..., 1]$.
  - Use `jnp.histogramdd` with sample `sample = jnp.stack([theta, phi], axis=-1)`.
  - Accumulate into `state["map"]` which will be a 2D array.

**Code Sketch**:
```python
class Density2DEstimator(Estimator[HallSystem]):
    observable_type = Density # Or custom type, but Density works if we just want raw map

    def __init__(self, adaptor, system, estimator_options, observable_options):
        super().__init__(adaptor, system, estimator_options, observable_options)
        self.bins_theta = self.options.get("bins_theta", 50)
        self.bins_phi = self.options.get("bins_phi", 100)
        # ...

    def evaluate(self, i, params, key, data, system, state, aux_data):
        theta = data[..., 0] 
        phi = data[..., 1] # Check phi range [-pi, pi]
        # Stack and histogram
        # ...
```

### 2.2 Task 2: Create Plotting Script `plot_disorder_density.py`
**File**: `data/plot_disorder_density.py` (or similar location)

This script will visualize the 2D density and overlay impurities.

**Steps**:
1.  **Load Data**:
    - Load `netobs_ckpt_*.npz` to get the 2D `state/map` counts.
    - Load `disorder_config.npz` to get impurity positions/charges.
2.  **Process Density**:
    - Convert counts to density $\rho(\theta, \phi)$.
    - Divide by appropriate area element (Jacobian $\sin\theta$).
    - Normalize such that $\int \rho d\Omega = N$.
3.  **Visualization**:
    - Use `plt.imshow`, `plt.pcolormesh`, or `plt.contourf`.
    - Axes: $\phi$ (x-axis) vs $\theta$ (y-axis).
    - Overlay impurities: scatter plot of impurity $(\phi, \theta)$ coords.
        - Distinguish positive/negative charges by color/marker.

### 2.3 Task 3: Integration
- The new estimator can be invoked via command line:
  `--observable_config "estimator=deephall@density2d"` (syntax depends on netobs CLI).
  Or simply by modifying the run script to use this estimator.
- We will assume standard `netobs` CLI usage is available.

---

## 3. Verification Strategy

1.  **Run with $N_{dis}=0$**:
    - Should recover uniform density (within noise).
    - $\rho(\theta, \phi) \approx \text{const}$.
2.  **Run with $N_{dis} > 0$**:
    - Observe peaks in $\rho(\theta, \phi)$ at impurity locations (if attractive) or valleys (if repulsive).
    - Check normalization integral $\approx N$.
