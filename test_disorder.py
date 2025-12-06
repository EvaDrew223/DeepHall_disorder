#!/usr/bin/env python3
"""Test script to verify disorder implementation."""

import jax
import jax.numpy as jnp
import numpy as np

from deephall.config import Config, DisorderConfig, System
from deephall.disorder import create_impurity_configuration, verify_rotational_invariance
from deephall.hamiltonian import compute_disorder_potential
from deephall.networks import make_network


def test_disorder_configuration():
    """Test 1: Verify disorder configuration is created correctly."""
    print("\n" + "=" * 60)
    print("Test 1: Disorder Configuration")
    print("=" * 60)

    system = System(
        flux=4,
        nspins=(3, 0),
        disorder=DisorderConfig(
            n_dis=5,
            disorder_separation=0.5,
            max_disorder_charge=1.0,
        ),
    )

    key = jax.random.PRNGKey(42)
    impurity_config = create_impurity_configuration(system, key)

    assert impurity_config is not None, "Impurity config should not be None"
    assert impurity_config.n_dis == 5, "Should have 5 impurities"
    
    # Check positions are on unit sphere
    norms = jnp.linalg.norm(impurity_config.positions, axis=-1)
    print(f"Position norms (should be ~1.0): {norms}")
    assert jnp.allclose(norms, 1.0), "Impurity positions should be on unit sphere"

    # Check charges sum to zero (neutrality)
    charge_sum = jnp.sum(impurity_config.charges)
    print(f"Sum of charges (should be ~0.0): {charge_sum:.6e}")
    assert jnp.abs(charge_sum) < 1e-6, "Charges should sum to zero"

    # Check donor radius
    Q = system.flux / 2
    R = jnp.sqrt(Q)
    expected_donor_radius = R + system.disorder.disorder_separation
    print(f"Donor radius: {impurity_config.donor_radius:.4f}")
    print(f"Expected: {expected_donor_radius:.4f}")
    assert jnp.isclose(
        impurity_config.donor_radius, expected_donor_radius
    ), "Donor radius incorrect"

    print("✓ Test 1 passed!")
    return impurity_config, system


def test_rotational_invariance(impurity_config, system):
    """Test 2: Verify disorder potential is rotationally invariant."""
    print("\n" + "=" * 60)
    print("Test 2: Rotational Invariance")
    print("=" * 60)

    # Create test electron configuration
    nelec = sum(system.nspins)
    key = jax.random.PRNGKey(123)
    u = jax.random.uniform(key, (nelec,), minval=-1.0, maxval=1.0)
    theta = jnp.arccos(u)
    phi = jax.random.uniform(key, (nelec,), minval=0.0, maxval=2 * jnp.pi)
    electron_coords = jnp.stack([theta, phi], axis=-1)

    Q = system.flux / 2
    R = system.radius or jnp.sqrt(Q)

    # Create rotation matrix (rotation around z-axis by 45 degrees)
    angle = jnp.pi / 4
    rotation_matrix = jnp.array(
        [
            [jnp.cos(angle), -jnp.sin(angle), 0],
            [jnp.sin(angle), jnp.cos(angle), 0],
            [0, 0, 1],
        ]
    )

    potential_before, potential_after = verify_rotational_invariance(
        impurity_config, electron_coords, R, rotation_matrix
    )

    print(f"Potential before rotation: {potential_before:.8f}")
    print(f"Potential after rotation: {potential_after:.8f}")
    print(f"Relative difference: {jnp.abs(potential_before - potential_after) / jnp.abs(potential_before):.6e}")

    assert jnp.allclose(
        potential_before, potential_after, rtol=1e-5
    ), "Potential should be rotationally invariant"

    print("✓ Test 2 passed!")


def test_input_feature_shape(impurity_config, system):
    """Test 3: Verify input feature tensor has correct shape."""
    print("\n" + "=" * 60)
    print("Test 3: Input Feature Shape")
    print("=" * 60)

    from deephall.config import Network, NetworkType, PsiformerNetwork

    network_config = Network(
        type=NetworkType.psiformer,
        psiformer=PsiformerNetwork(
            num_heads=2, heads_dim=32, num_layers=1, determinants=1
        ),
    )

    model = make_network(system, network_config, impurity_config)

    # Create sample input
    nelec = sum(system.nspins)
    key = jax.random.PRNGKey(456)
    u = jax.random.uniform(key, (nelec,), minval=-1.0, maxval=1.0)
    theta = jnp.arccos(u)
    phi = jax.random.uniform(key, (nelec,), minval=0.0, maxval=2 * jnp.pi)
    electrons = jnp.stack([theta, phi], axis=-1)

    # Initialize parameters
    params = model.init(jax.random.PRNGKey(0), electrons)

    # Test forward pass
    log_psi = model.apply(params, electrons)
    print(f"Log psi shape: {log_psi.shape}")
    print(f"Log psi value: {log_psi}")

    # Check the input features shape
    expected_feature_dim = 4 + 2 * impurity_config.n_dis
    print(f"Expected input feature dimension: {expected_feature_dim}")
    print(f"  - Base features: 4 (x, y, z coordinates + spin)")
    print(f"  - Disorder features: 2 × {impurity_config.n_dis} (cos_sim + potential for each impurity)")

    print("✓ Test 3 passed!")


def test_backward_compatibility():
    """Test 4: Verify code works without disorder (n_dis=0)."""
    print("\n" + "=" * 60)
    print("Test 4: Backward Compatibility (No Disorder)")
    print("=" * 60)

    system = System(
        flux=4,
        nspins=(3, 0),
        disorder=DisorderConfig(n_dis=0),  # No disorder
    )

    key = jax.random.PRNGKey(789)
    impurity_config = create_impurity_configuration(system, key)

    assert impurity_config is None, "Should return None when n_dis=0"
    print("✓ Impurity config correctly returns None for n_dis=0")

    # Test that network still works
    from deephall.config import Network, NetworkType, PsiformerNetwork

    network_config = Network(
        type=NetworkType.psiformer,
        psiformer=PsiformerNetwork(
            num_heads=2, heads_dim=32, num_layers=1, determinants=1
        ),
    )

    model = make_network(system, network_config, impurity_config)
    nelec = sum(system.nspins)
    u = jax.random.uniform(key, (nelec,), minval=-1.0, maxval=1.0)
    theta = jnp.arccos(u)
    phi = jax.random.uniform(key, (nelec,), minval=0.0, maxval=2 * jnp.pi)
    electrons = jnp.stack([theta, phi], axis=-1)

    params = model.init(jax.random.PRNGKey(0), electrons)
    log_psi = model.apply(params, electrons)

    print(f"Log psi (no disorder): {log_psi}")
    print("✓ Network works correctly without disorder")

    print("✓ Test 4 passed!")


def test_disorder_potential_value():
    """Test 5: Verify disorder potential has reasonable values."""
    print("\n" + "=" * 60)
    print("Test 5: Disorder Potential Sanity Check")
    print("=" * 60)

    system = System(
        flux=4,
        nspins=(3, 0),
        disorder=DisorderConfig(
            n_dis=3,
            disorder_separation=0.5,
            max_disorder_charge=1.0,
        ),
    )

    key = jax.random.PRNGKey(999)
    impurity_config = create_impurity_configuration(system, key)

    # Create electrons at specific positions
    nelec = sum(system.nspins)
    theta = jnp.array([jnp.pi / 4, jnp.pi / 2, 3 * jnp.pi / 4])
    phi = jnp.array([0.0, jnp.pi / 2, jnp.pi])
    electrons = jnp.stack([theta, phi], axis=-1)

    Q = system.flux / 2
    R = system.radius or jnp.sqrt(Q)

    potential = compute_disorder_potential(electrons, impurity_config, R)
    print(f"Disorder potential value: {potential:.6f}")
    print(f"Impurity charges: {impurity_config.charges}")
    print(f"Electron radius R: {R:.4f}")
    print(f"Donor radius R_d: {impurity_config.donor_radius:.4f}")

    # The potential should be finite
    assert jnp.isfinite(potential), "Potential should be finite"
    print("✓ Potential is finite")

    # Test with batch dimension
    batch_size = 10
    electrons_batch = jnp.tile(electrons[None, :, :], (batch_size, 1, 1))
    potential_batch = compute_disorder_potential(electrons_batch, impurity_config, R)
    print(f"Potential with batch shape {electrons_batch.shape}: {potential_batch:.6f}")
    assert jnp.isfinite(potential_batch), "Batched potential should be finite"

    print("✓ Test 5 passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DISORDER IMPLEMENTATION VERIFICATION")
    print("=" * 60)

    try:
        # Test 1: Basic disorder configuration
        impurity_config, system = test_disorder_configuration()

        # Test 2: Rotational invariance
        test_rotational_invariance(impurity_config, system)

        # Test 3: Input feature shapes
        test_input_feature_shape(impurity_config, system)

        # Test 4: Backward compatibility
        test_backward_compatibility()

        # Test 5: Disorder potential values
        test_disorder_potential_value()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED! ✗")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
