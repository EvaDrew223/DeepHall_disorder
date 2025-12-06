# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This code is an reimplementation of the Psiformer network (Glehn et al., ICLR 2023).

The input feature is chosen as the Cartesian coordinates of the elelctrons:
    h_one_0 = [cos(theta), sin(theta) cos(phi), sin(theta) sin(phi)]
The feature is then passed through standard Psiformer layers, outputing features h_one.
Afterwards, the features are used to construct the orbitals with the monopole harmonics.
The details for the orbital construction are located in `blocks.py`.
"""

from flax import linen as nn
from jax import numpy as jnp

from deephall.config import OrbitalType
from deephall.types import ImpurityConfiguration

from .blocks import Jastrow, Orbitals


class PsiformerLayers(nn.Module):
    num_heads: int
    heads_dim: int
    num_layers: int
    impurity_config: ImpurityConfiguration | None
    sphere_radius: float

    @nn.compact
    def __call__(self, electrons: jnp.ndarray, spins: jnp.ndarray):
        theta, phi = electrons[..., 0], electrons[..., 1]
        h_one = self.input_feature(theta, phi, spins)
        attention_dim = self.num_heads * self.heads_dim
        h_one = nn.Dense(attention_dim, use_bias=False)(h_one)
        for _ in range(self.num_layers):
            attn_out = nn.MultiHeadAttention(num_heads=self.num_heads)(h_one)
            h_one += nn.Dense(attention_dim, use_bias=False)(attn_out)
            h_one = nn.LayerNorm(epsilon=1e-5)(h_one)
            h_one += nn.tanh(nn.Dense(attention_dim)(h_one))
            h_one = nn.LayerNorm(epsilon=1e-5)(h_one)
        return h_one

    def input_feature(self, theta: jnp.ndarray, phi: jnp.ndarray, spins: jnp.ndarray):
        """Generate input features for electrons, augmented with disorder information.

        Args:
            theta: Polar angles of electrons, shape (..., nelec).
            phi: Azimuthal angles of electrons, shape (..., nelec).
            spins: Spin quantum numbers, shape (nelec,).

        Returns:
            Feature tensor of shape (..., nelec, feature_dim).
            - feature_dim = 4 (clean) or 4 + 2*n_dis (with disorder)
        """
        # Base features: Cartesian coordinates + spin
        base_features = jnp.stack(
            [
                jnp.cos(theta),
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                spins,
            ],
            axis=-1,
        )  # shape: (..., nelec, 4)

        # If no disorder, return base features
        if self.impurity_config is None or self.impurity_config.n_dis == 0:
            return base_features

        # Add disorder features
        # Convert electron positions to Cartesian
        xyz_e = jnp.stack(
            [
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                jnp.cos(theta),
            ],
            axis=-1,
        )  # shape: (..., nelec, 3)

        # Impurity positions (Cartesian on unit sphere)
        xyz_imp = self.impurity_config.positions  # shape: (n_dis, 3)

        # Cosine similarity: c_iI = n_i · n_I
        cos_similarity = jnp.einsum(
            "...ic,ac->...ia", xyz_e, xyz_imp
        )  # (..., nelec, n_dis)

        # Local potential contribution: V_iI = z_I / |r_i - R_I|
        R_d = self.impurity_config.donor_radius
        r = self.sphere_radius
        distance = jnp.sqrt(r**2 + R_d**2 - 2 * r * R_d * cos_similarity)
        charges = self.impurity_config.charges  # shape: (n_dis,)
        local_potential = charges / distance / r  # (..., nelec, n_dis)

        # Interleave cosine similarity and potential for each impurity
        # Shape: (..., nelec, n_dis, 2)
        disorder_features = jnp.stack([cos_similarity, local_potential], axis=-1)
        # Reshape to (..., nelec, 2*n_dis)
        disorder_features = disorder_features.reshape(
            *disorder_features.shape[:-2], -1
        )

        # Concatenate base and disorder features
        return jnp.concatenate([base_features, disorder_features], axis=-1)


class Psiformer(nn.Module):
    nspins: tuple[int, int]
    Q: float
    ndets: int
    num_heads: int
    heads_dim: int
    num_layers: int
    orbital_type: OrbitalType
    impurity_config: ImpurityConfiguration | None = None
    sphere_radius: float = 1.0

    def __call__(self, electrons):
        orbitals = self.orbitals(electrons)
        signs, logdets = jnp.linalg.slogdet(orbitals)
        logmax = jnp.max(logdets)  # logsumexp trick
        return jnp.log(jnp.sum(signs * jnp.exp(logdets - logmax))) + logmax

    @nn.compact
    def orbitals(self, electrons):
        theta, phi = electrons[..., 0], electrons[..., 1]
        spins = jnp.array([1] * self.nspins[0] + [-1] * self.nspins[1])
        h_one = PsiformerLayers(
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            heads_dim=self.heads_dim,
            impurity_config=self.impurity_config,
            sphere_radius=self.sphere_radius,
        )(electrons, spins)
        orbitals = Orbitals(
            type=self.orbital_type, Q=self.Q, nspins=self.nspins, ndets=self.ndets
        )(h_one, theta, phi)
        jastrow = Jastrow(self.nspins)(electrons)
        return jnp.exp(jastrow / sum(self.nspins)) * orbitals
