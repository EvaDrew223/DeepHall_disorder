# Disorder Extension for DeepHall

## Overview

This document provides a comprehensive guide to the random charge disorder implementation in the DeepHall codebase. This extension adds support for studying phase transitions in the Fractional Quantum Hall (FQH) effect with disorder, including Wigner crystallization, charge density waves, and Anderson localization.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Physics Background](#physics-background)
3. [Implementation Summary](#implementation-summary)
4. [Key Features](#key-features)
5. [Usage Guide](#usage-guide)
6. [Files Modified](#files-modified)
7. [Detailed Implementation](#detailed-implementation)
8. [Design Decisions](#design-decisions)
9. [Verification & Testing](#verification--testing)
10. [Troubleshooting](#troubleshooting)
11. [Future Extensions](#future-extensions)

---

## Quick Start

### 1. Test the Implementation

```bash
python test_disorder.py
```

Expected output: `ALL TESTS PASSED! ✓`

### 2. Run with Disorder

```bash
# Simple example with 10 impurities
python -m deephall.train \
    system.disorder.n_dis=10 \
    system.disorder.disorder_separation=0.5

# Or use the example config file
python -m deephall.train --yml example_disorder_config.yml
```

### 3. Run without Disorder (Original Behavior)

```bash
python -m deephall.train system.disorder.n_dis=0
```

---

## Physics Background

### System Geometry

- **N electrons** on a sphere of radius $R = \sqrt{Q} \cdot \ell$
- **N_dis impurities** with charges $\{q_a\}$ on a concentric donor shell at radius $R_d$
- **Separation d** between electron sphere and donor shell: $R_d = R + d$

### Disorder Potential

The disorder potential acting on electrons is:

$$V_{\text{dis}}(\Omega_i) = \sum_a \frac{z_a}{|r_i - R_a|} / R$$

where:
- $z_a = q_a/e$ (impurity charge in units of $e$)
- $|r_i - R_a| = \sqrt{R^2 + R_d^2 - 2RR_d \cdot \cos(\gamma_{ia})}$
- $\cos(\gamma_{ia}) = \mathbf{n}_i \cdot \mathbf{n}_a$ (dot product of unit vectors)
- $\sum_a q_a = 0$ (neutrality constraint)

The energy is normalized by $1/R$ to match the units of the electron-electron Coulomb interaction in original DeepHall code.

### Neural Network Input Features

**Clean system (original)**: For electron $i$, input features are 4D:
$$f_i = [\cos(\theta), \sin(\theta)\cos(\phi), \sin(\theta)\sin(\phi), \text{spin}]$$

**With disorder**: For electron $i$, input features are $(4 + 2 \cdot N_{\text{dis}})$ D:
$$f_i = [\cos(\theta), \sin(\theta)\cos(\phi), \sin(\theta)\sin(\phi), \text{spin}, c_{i1}, V_{i1}, c_{i2}, V_{i2}, \ldots, c_{iN_{\text{dis}}}, V_{iN_{\text{dis}}}]$$

where for each impurity $I$:
- $c_{iI} = \mathbf{n}_i \cdot \mathbf{n}_I$ (cosine similarity)
- $V_{iI} = z_I / |r_i - R_I| / R$ (local potential contribution)

---

## Implementation Summary
#### 1. Disorder Configuration
- **File**: `deephall/disorder.py` (new)
- **File**: `deephall/config.py` (modified)
- **File**: `deephall/types.py` (modified)

Created `DisorderConfig` with parameters:
- `n_dis`: Number of impurities
- `disorder_separation`: Distance d between electron and donor shells
- `max_disorder_charge`: Maximum charge magnitude
- `disorder_strength`: Overall scaling factor (0.0 to 1.0)
- `disorder_seed`: Random seed for reproducibility

Impurities are:
- Randomly distributed on unit sphere
- Assigned charges with equal magnitude but random signs (±)
- Automatically adjusted to ensure neutrality ($\sum q = 0$)

#### 2. Hamiltonian Update
- **File**: `deephall/hamiltonian.py` (modified)

Implemented `compute_disorder_potential()` function:
- Vectorized over batch of walkers
- Computes: $V_{\text{dis}} = \sum_i \sum_a (z_a / |r_i - R_a|) / R$
- Uses same energy normalization as Coulomb interaction
- Integrated into `local_energy()` function

#### 3. Input Feature Update
- **File**: `deephall/networks/psiformer.py` (modified)
- **File**: `deephall/networks/__init__.py` (modified)

Modified `PsiformerLayers.input_feature()` to augment electron features:
- **Original**: 4D per electron
- **With disorder**: $(4 + 2 \cdot N_{\text{dis}})$ D per electron
- **Added features**: For each impurity: $[c_{iI}, V_{iI}]$

The network architecture automatically adapts to the new input dimension.

#### 4. Verification
- **File**: `test_disorder.py` (new)

Comprehensive test suite with 5 tests:
1. ✅ Disorder configuration creation and validation
2. ✅ Rotational invariance verification
3. ✅ Input feature tensor shape validation
4. ✅ Backward compatibility ($n_{\text{dis}}=0$ works)
5. ✅ Disorder potential sanity checks

All tests pass successfully.

---

## Key Features

### Disorder Persistence
Disorder is generated once at initialization (using `disorder_seed` if provided, otherwise system seed) and kept fixed throughout training. The configuration is saved to `disorder_config.npz`.

### Energy Units
The disorder potential uses the same energy normalization as the electron-electron Coulomb interaction:
- Coulomb: $V_{\text{ee}} = \sum_{i<j} (1/r_{ij}) / R$
- Disorder: $V_{\text{dis}} = \sum_i \sum_a (q_a/|r_i - R_a|) / R$

Both divide by radius R, ensuring consistent energy units.

### Charge Distribution
Charges have equal magnitude (`max_disorder_charge`) but random signs (±1). They are then adjusted by subtracting the mean to ensure exact neutrality ($\sum q = 0$).

### Input Feature Dimensions
- **Original 4 features**: $[\cos(\theta), \sin(\theta)\cos(\phi), \sin(\theta)\sin(\phi), \text{spin}]$
  - First 3 are Cartesian coordinates on unit sphere
  - Last is spin quantum number
- **Added $2 \cdot N_{\text{dis}}$ features**: For each impurity $I$: $[c_{iI}, V_{iI}]$
- **Total**: $4 + 2 \cdot N_{\text{dis}}$ features per electron

For small $N_{\text{dis}}$ (<20), explicit concatenation is implemented.

### Backward Compatibility
Setting `n_dis=0` completely disables disorder:
- `create_impurity_configuration()` returns `None`
- All code gracefully handles `impurity_config=None`
- Network falls back to original 4D input features
- Test suite verifies this works correctly

---

## Usage Guide

### Key Parameters

- **`n_dis`**: Number of impurities (0 to disable disorder)
- **`disorder_separation`**: Distance between electron sphere and donor shell
- **`max_disorder_charge`**: Maximum magnitude of impurity charges
- **`disorder_strength`**: Overall strength scaling (0.0 to 1.0)
- **`disorder_seed`**: Random seed for reproducibility

### Basic Python Example

```python
from deephall.config import Config, DisorderConfig, System

# Configure system with disorder
config = Config(
    system=System(
        flux=4,
        nspins=(3, 0),
        disorder=DisorderConfig(
            n_dis=10,                    # 10 impurities
            disorder_separation=0.5,     # d = 0.5
            max_disorder_charge=1.0,     # |q| ≤ 1.0
            disorder_strength=1.0,       # Full strength
            disorder_seed=42,            # For reproducibility
        ),
    ),
)
```

### Command Line Usage

```bash
# Run with disorder
python -m deephall.train \
    system.disorder.n_dis=10 \
    system.disorder.disorder_separation=0.5 \
    system.disorder.max_disorder_charge=1.0 \
    system.disorder.disorder_strength=1.0

# Run without disorder (backward compatible)
python -m deephall.train \
    system.disorder.n_dis=0

# Alternative: direct file execution
python deephall/train.py system.disorder.n_dis=10
```

### Output Files

When running with disorder, you'll get:
- `disorder_config.npz` - Saved impurity configuration
- `train_stats.csv` - Includes `disorder_potential` column
- `config.yml` - Includes disorder settings

---

## Files Modified

### Core Files (Modified - 9 files)

1. **`deephall/config.py`**
   - Added `DisorderConfig` dataclass with parameters
   - Added `disorder` field to `System` config

2. **`deephall/types.py`**
   - Added `ImpurityConfiguration` NamedTuple
   - Added `disorder_potential` to `OtherObservables`

3. **`deephall/hamiltonian.py`**
   - Added `compute_disorder_potential()` function
   - Modified `local_energy()` to accept and use impurity_config
   - Disorder potential is added to total energy

4. **`deephall/networks/psiformer.py`**
   - Modified `PsiformerLayers` to accept `impurity_config` and `sphere_radius`
   - Updated `input_feature()` method to augment with disorder features
   - Modified `Psiformer` class to pass disorder config to layers

5. **`deephall/networks/__init__.py`**
   - Modified `make_network()` to accept and pass impurity_config

6. **`deephall/loss.py`**
   - Modified `make_loss_fn()` to accept and pass impurity_config

7. **`deephall/optimizers/__init__.py`**
   - Modified `make_optimizer_step()` to accept and pass impurity_config

8. **`deephall/train.py`**
   - Added disorder configuration creation at initialization
   - Disorder config is logged and saved
   - `disorder_potential` is logged in training statistics

9. **`deephall/log.py`**
   - Added `save_disorder_config()` method to save disorder to file

### New Files (3 files)

1. **`deephall/disorder.py`**
   - `create_impurity_configuration()`: Generate random disorder config
   - `verify_rotational_invariance()`: Test rotational invariance

2. **`test_disorder.py`**
   - Comprehensive test suite for disorder implementation
   - Tests: configuration, rotational invariance, feature shapes, backward compatibility, potential values

3. **`DISORDER_IMPLEMENTATION.md`** (comprehensive documentation)

---

## Detailed Implementation

### Disorder Configuration Creation

The `create_impurity_configuration()` function:
1. Generates random positions on unit sphere
2. Assigns charges with equal magnitude but random signs
3. Adjusts charges to ensure exact neutrality
4. Returns `ImpurityConfiguration` object

### Disorder Potential Computation

The `compute_disorder_potential()` function:
1. Takes electron coordinates and impurity configuration
2. Computes distances between electrons and impurities
3. Calculates potential for each electron
4. Normalizes by sphere radius R
5. Returns potential array of shape (batch_size, n_electrons)

### Network Input Augmentation

The `input_feature()` method:
1. Computes original 4D features from electron positions
2. For each impurity:
   - Computes cosine similarity $c_{iI} = \mathbf{n}_i \cdot \mathbf{n}_I$
   - Computes local potential $V_{iI}$
3. Concatenates all features into single vector
4. Returns augmented feature tensor

### Integration Through Pipeline

```
train.py creates disorder config
    ↓
config.yml includes disorder settings
    ↓
make_network() receives impurity_config
    ↓
Psiformer augments input features
    ↓
hamiltonian computes disorder potential
    ↓
loss function evaluates energy
    ↓
optimizer performs gradient descent
    ↓
train_stats.csv logs disorder_potential
    ↓
disorder_config.npz saves configuration
```

---

## Design Decisions

1. **Explicit concatenation**: For small $N_{\text{dis}}$ (<20), we explicitly concatenate disorder features for each impurity. This allows the network to learn individual impurity-electron interactions.

2. **Equal magnitude charges**: Charges have equal $|q|$ but random signs (±). This simplifies the disorder while maintaining physical realism.

3. **Backward compatibility**: Setting `n_dis=0` disables disorder completely, returning `None` for impurity_config. The code gracefully handles this everywhere.

4. **Fixed disorder**: Disorder is generated once at initialization and kept fixed throughout training. This simulates a quenched disorder scenario relevant for phase transitions.

5. **Energy normalization**: Division by R ensures disorder potential has the same energy scale as Coulomb interaction.

6. **Neutral impurities**: Charges are adjusted to ensure $\sum_a q_a = 0$ exactly, preventing spurious electrostatic effects.

---

## Verification & Testing

### Run the Test Suite

```bash
python test_disorder.py
```

Expected output:
```
============================================================
ALL TESTS PASSED! ✓
============================================================
```

### Tests Included

1. **Disorder Configuration Test**
   - Verifies disorder configuration creation
   - Checks neutrality constraint ($\sum q = 0$)
   - Verifies positions are on unit sphere

2. **Rotational Invariance Test**
   - Verifies potential unchanged under rotation
   - Tests with multiple random rotations

3. **Input Feature Shape Test**
   - Checks correct feature dimensions
   - Verifies $(4 + 2 \cdot N_{\text{dis}})$D output

4. **Backward Compatibility Test**
   - Runs with `n_dis=0`
   - Confirms impurity_config is `None`
   - Verifies network still works

5. **Potential Sanity Check**
   - Ensures all potential values are finite
   - Checks for NaN or Inf

### Physics Validation

#### Rotational Invariance
The disorder potential $V_{\text{dis}}$ depends only on relative positions between electrons and impurities. When the entire system (electrons + impurities) is rotated together, the potential remains unchanged. This is verified in the test suite.

#### Energy Units
- Electron-electron Coulomb: $V_{\text{ee}} = \sum_{i<j} 1/r_{ij} / R$
- Disorder potential: $V_{\text{dis}} = \sum_i \sum_a q_a/|r_i - R_a| / R$

Both are normalized by $1/R$, ensuring consistent energy units.

#### Neutrality
The impurity charges are generated with equal magnitudes but random signs, then adjusted to ensure $\sum_a q_a = 0$ exactly. This prevents spurious electrostatic effects.

---

## Troubleshooting

### Issue: NaN in energy

**Possible causes:**
- Impurities too close to electron sphere (d too small)
- Disorder strength too large

**Solutions:**
- Increase `disorder_separation` (e.g., d=0.5 or larger)
- Reduce `disorder_strength` or `max_disorder_charge`

### Issue: Network not converging

**Possible causes:**
- Too many impurities (high-dimensional input)
- Disorder too strong

**Solutions:**
- Start with small `n_dis` (e.g., 5-10)
- Gradually increase disorder_strength from 0 to 1
- Adjust learning rate or network architecture

### Issue: Memory errors

**Possible causes:**
- Large $N_{\text{dis}}$ increases memory for input features

**Solutions:**
- Reduce batch_size
- Reduce n_dis
- Consider implementing implicit disorder encoding

---

## Future Extensions

Possible extensions not yet implemented:

1. **Variable charge magnitudes**: Allow different $|q_a|$ for each impurity
2. **Correlation in disorder**: Generate spatially correlated disorder patterns
3. **Disorder averaging**: Average over multiple disorder realizations
4. **Implicit disorder encoding**: For large $N_{\text{dis}}$, use attention-based encoding instead of explicit concatenation
5. **Time-dependent disorder**: Allow disorder to change during training (annealing)
6. **Disorder strength annealing**: Gradually increase disorder during training

---

## How to Use the Disorder Feature

### Example 1: Run with Small Disorder

```bash
python -m deephall.train \
    system.disorder.n_dis=5 \
    system.disorder.disorder_separation=0.5 \
    system.disorder.disorder_strength=0.5
```

### Example 2: Run with Strong Disorder

```bash
python -m deephall.train \
    system.disorder.n_dis=10 \
    system.disorder.disorder_separation=0.5 \
    system.disorder.max_disorder_charge=1.0 \
    system.disorder.disorder_strength=1.0
```

### Example 3: Using Example Config

```bash
python -m deephall.train --yml example_disorder_config.yml
```

### Example 4: Reproducible Run

```bash
python -m deephall.train \
    system.disorder.n_dis=10 \
    system.disorder.disorder_separation=0.5 \
    system.disorder.disorder_seed=42
```

---

## Example Configuration File

See `example_disorder_config.yml` for a complete example configuration with disorder settings.

---

## Documentation Files

- **`DISORDER_IMPLEMENTATION.md`**: Complete technical documentation
- **`DISORDER_README.md`**: Quick start guide
- **`IMPLEMENTATION_SUMMARY.md`**: Implementation overview
- **`example_disorder_config.yml`**: Example configuration file
- **`test_disorder.py`**: Test suite
- **`prompt_disorder.md`**: Original requirements (if available)

---

## Summary

The disorder implementation provides:
- ✅ Random charge disorder on donor shell
- ✅ Modified Hamiltonian with disorder potential
- ✅ Augmented neural network input features
- ✅ Full backward compatibility
- ✅ Comprehensive test suite
- ✅ Reproducible disorder generation
- ✅ Consistent energy normalization
- ✅ Rotational invariance

You can now study phase transitions in the Fractional Quantum Hall effect with disorder!

---

## References

1. DeepHall original repository
2. Psiformer architecture: Glehn et al., ICLR 2023
3. Physics background: "Composite Fermions" textbook
4. Original disorder prompt: `prompt_disorder.md`

---

## Contact & Support

For questions or issues with the disorder implementation:
1. Check this documentation
2. Run the test suite: `python test_disorder.py`
3. Review the original implementation guide in `DISORDER_IMPLEMENTATION.md`
4. Check example configuration in `example_disorder_config.yml`
