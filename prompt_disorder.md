# Task: Extend DeepHall to Include Random Charge Disorder

**Role:** You are an expert Computational Physicist and Machine Learning Engineer specializing in Quantum Monte Carlo (QMC) and Neural Quantum States.

**Context:**
I have an existing codebase ("DeepHall") that uses a Transformer-based neural network (Psiformer architecture) to solve the Fractional Quantum Hall (FQH) effect on a sphere. Currently, the code handles the clean system.
**My goal is to introduce random charge disorders to the system to study phase transitions (Wigner crystallization, charge density wave, Anderson localization, etc.).**

I need you to modify the existing code to include:
1.  A random charge disorder potential in the Hamiltonian.
2.  Modified input features for the neural network to encode the disorder landscape ("positional encoding" for the impurities).

Below are the detailed physical specifications and implementation steps.

---

## 1. Physics Specifications

### The Geometry
* **System:** $N$ electrons on a sphere of radius $R = \sqrt{Q}\ell$.
* **Disorder:** $N_{dis}$ fixed point charges $\{q_a\}$ located on a concentric "donor shell" at radius $R_d$.
* **Separation:** The distance between the electron sphere and disorder shell is $d = R_d - R$. Assume $d \neq 0$ to avoid singularities (cusps).

### The Hamiltonian
The total Hamiltonian is:
$$\hat{H} = \hat{H}_{clean} + \hat{V}_{dis}$$
where the disorder potential acting on the electrons is:
$$\hat{V}_{dis} = \sum_{i=1}^N V_{dis}(\Omega_i)$$

The single-particle potential $V_{dis}$ felt by an electron at $\Omega_i$ due to impurities at $\Omega_a$ is:
$$V_{dis}(\Omega_i) = \frac{e^2}{\epsilon} \sum_{a=1}^{N_{dis}} \frac{z_a}{|\mathbf{r}_i - \mathbf{R}_a|} = \frac{e^2}{\epsilon} \sum_{a=1}^{N_{dis}} \frac{z_a}{\sqrt{R^2 + R_d^2 - 2RR_d\cos \gamma_{ia}}}$$
* $z_a = q_a/e$ (impurity strength).
* $\cos \gamma_{ia} = \mathbf{n}_i \cdot \mathbf{n}_a$ (dot product of unit vectors).
* Constraint: $\sum_a q_a = 0$ (neutral background).

---

## 2. Neural Network Architecture Modifications

The current DeepHall architecture uses a Psiformer. The inputs are electron coordinates transformed into embeddings, processed by self-attention, and output into a determinant.

**Current Input Features:**
For electron $i$, the feature vector is Cartesian coordinates on the unit sphere:
$$f_i^0 = [\sin\theta_i\cos\phi_i, \sin\theta_i\sin\phi_i, \cos\theta_i]$$

**New Requirement:**
Disorder breaks translational symmetry. The network must "see" the potential landscape. You must implement a feature generator that augments $f_i^0$ with disorder information. For low $N_{dis}$, you explicitly implement impurity features by concatenating the relationship between electron $i$ and *every* impurity $I$:
$$f_i^{new} = \text{concat}(f_i^0, [c_{i1}, V_{i1}], [c_{i2}, V_{i2}], \dots, [c_{i N_{dis}}, V_{i N_{dis}}])$$
* $c_{iI} = \mathbf{n}_i \cdot \mathbf{n}_I$ (cosine similarity).
* $V_{iI} \propto q_I / |\mathbf{r}_i - \mathbf{R}_I|$ (local potential contribution).

---

## 3. Implementation Plan

Please analyze the code in folder "deephall" in the current repo, and perform the following changes. Do not change the core Transformer/Attention logic, only the Inputs and the Hamiltonian (and therefore also the local energy in VMC part).

### Step 1: Disorder Configuration
Create a new module or class `DisorderConfig` that:
1.  Initializes random positions $\{\Omega_a\}$ for $N_{dis}$ impurities on the sphere, set $N_{dis}$ and maximum magnitude of disorder charge max($|q|$) as input by the user.
2.  Assigns charges $\{q_a\}$ (ensure sum is 0).
3.  Stores $R_d$ (based on parameter $d$).

### Step 2: Hamiltonian Update
In the local energy calculation function (likely in `hamiltonian.py` or `vmc.py`):
1.  Implement the function `compute_disorder_potential(electron_coords, impurity_config)`.
2.  Add this scalar result to the existing Coulomb and Kinetic energy terms.
3.  Ensure the calculation is vectorized over the batch of walkers.

### Step 3: Input Feature Update
In the neural network definition (likely `network.py` or `model.py`):
1.  Locate the initial embedding layer where `f_i^0` is constructed.
2.  Inject the `impurity_config` into the model's `apply` or `forward` method.
3.  Implement the logic for $f^{new}_i$ described above.
4.  Adjust the dimension of the first linear projection layer ($W^0$) to match the new input size.

### Step 4: Verification
Provide a small test snippet to:
1.  Verify that $V_{dis}$ is rotationally invariant if the impurities are rotated along with electrons.
2.  Check that the input feature tensor shape is correct: `(Batch, N_electrons, Feature_Dim)`.

---

**Codebase Constraints:**
* Maintain the existing coding style.
* Use vectorized operations (no explicit for-loops over particles if possible).

**Action:**
Please review the code files in folder "deephall", specifically looking for the Hamiltonian definition and the Input Embedding layer. Then, generate the code blocks to implement the changes above.