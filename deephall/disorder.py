"""Utilities for generating and managing disorder configurations."""

import jax
import jax.numpy as jnp
from chex import PRNGKey

from deephall.config import System
from deephall.types import ImpurityConfiguration


def create_impurity_configuration(
    system: System, key: PRNGKey
) -> ImpurityConfiguration | None:
    """Create a fixed impurity configuration from system parameters.

    Args:
        system: System configuration containing disorder parameters.
        key: JAX random key for reproducibility.

    Returns:
        ImpurityConfiguration if disorder is present, None otherwise.
    """
    disorder = system.disorder
    if disorder.n_dis == 0:
        return None

    # Split random key for positions and charges
    key_pos, key_charges = jax.random.split(key)

    # Generate random positions on unit sphere
    # Use uniform sampling: theta from arccos(uniform(-1,1)), phi from uniform(0, 2pi)
    n_dis = disorder.n_dis
    u = jax.random.uniform(key_pos, (n_dis,), minval=-1.0, maxval=1.0)
    theta_imp = jnp.arccos(u)
    phi_imp = jax.random.uniform(key_pos, (n_dis,), minval=0.0, maxval=2 * jnp.pi)

    # Convert to Cartesian coordinates on unit sphere
    positions = jnp.stack(
        [
            jnp.sin(theta_imp) * jnp.cos(phi_imp),
            jnp.sin(theta_imp) * jnp.sin(phi_imp),
            jnp.cos(theta_imp),
        ],
        axis=-1,
    )  # shape: (n_dis, 3)

    # Generate charges with equal magnitude but random signs
    signs = jax.random.choice(key_charges, jnp.array([-1.0, 1.0]), shape=(n_dis,))
    charges = signs * disorder.max_disorder_charge

    # Ensure neutrality: adjust charges to sum to exactly zero
    # Subtract mean from all charges
    charges = charges - jnp.mean(charges)

    # Calculate donor shell radius: R_d = R + d
    Q = system.flux / 2
    electron_radius = system.radius or jnp.sqrt(Q)
    donor_radius = electron_radius + disorder.disorder_separation

    return ImpurityConfiguration(
        n_dis=n_dis,
        positions=positions,
        charges=charges,
        donor_radius=donor_radius,
    )


def verify_rotational_invariance(
    impurity_config: ImpurityConfiguration,
    electron_coords: jnp.ndarray,
    sphere_radius: float,
    rotation_matrix: jnp.ndarray,
) -> tuple[float, float]:
    """Verify that the disorder potential is rotationally invariant.

    Apply the same rotation to both electrons and impurities, and verify that
    the potential energy is unchanged.

    Args:
        impurity_config: Configuration of impurities.
        electron_coords: Electron coordinates (theta, phi) of shape (nelec, 2).
        sphere_radius: Radius of electron sphere.
        rotation_matrix: 3x3 rotation matrix to apply.

    Returns:
        Tuple of (potential_before, potential_after) for comparison.
    """
    from deephall.hamiltonian import compute_disorder_potential

    # Compute potential before rotation
    potential_before = compute_disorder_potential(
        electron_coords, impurity_config, sphere_radius
    )

    # Convert electron coords to Cartesian
    theta_e, phi_e = electron_coords[..., 0], electron_coords[..., 1]
    xyz_e = jnp.stack(
        [
            jnp.sin(theta_e) * jnp.cos(phi_e),
            jnp.sin(theta_e) * jnp.sin(phi_e),
            jnp.cos(theta_e),
        ],
        axis=-1,
    )

    # Rotate electrons
    xyz_e_rot = jnp.dot(xyz_e, rotation_matrix.T)

    # Rotate impurities
    xyz_imp_rot = jnp.dot(impurity_config.positions, rotation_matrix.T)

    # Convert rotated electrons back to spherical
    x, y, z = xyz_e_rot[..., 0], xyz_e_rot[..., 1], xyz_e_rot[..., 2]
    theta_e_rot = jnp.arccos(z)
    phi_e_rot = jnp.arctan2(y, x)
    electron_coords_rot = jnp.stack([theta_e_rot, phi_e_rot], axis=-1)

    # Create rotated impurity config
    impurity_config_rot = ImpurityConfiguration(
        n_dis=impurity_config.n_dis,
        positions=xyz_imp_rot,
        charges=impurity_config.charges,
        donor_radius=impurity_config.donor_radius,
    )

    # Compute potential after rotation
    potential_after = compute_disorder_potential(
        electron_coords_rot, impurity_config_rot, sphere_radius
    )

    return potential_before, potential_after
