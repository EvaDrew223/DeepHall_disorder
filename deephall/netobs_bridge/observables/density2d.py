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

from typing import Any

from jax import numpy as jnp
from netobs.observables import Estimator
from netobs.observables.density import Density

from deephall.netobs_bridge.hall_system import HallSystem


class Density2DEstimator(Estimator[HallSystem]):
    observable_type = Density

    def __init__(self, adaptor, system, estimator_options, observable_options):
        super().__init__(adaptor, system, estimator_options, observable_options)
        # Bins can be a single int or a tuple (n_theta, n_phi)
        self.bins = self.options.get("bins", (50, 100))
        
        # Ensure bins is a tuple/list for 2D if it's an int
        if isinstance(self.bins, int):
            self.bins = (self.bins, self.bins * 2)

    def empty_val_state(
        self, steps: int
    ) -> tuple[dict[str, jnp.ndarray], dict[str, Any]]:
        del steps
        # Determine shape for zero array
        if isinstance(self.bins, int):
            shape = (self.bins, self.bins)
        else:
            shape = tuple(self.bins)
            
        return {}, {"map": jnp.zeros(shape)}

    def evaluate(
        self, i, params, key, data, system, state, aux_data
    ) -> tuple[dict[str, jnp.ndarray], dict[str, Any]]:
        del i, params, system, aux_data, key
        
        # Flatten batch and electron dimensions
        theta_flat = jnp.reshape(data[..., 0], (-1,))
        phi_flat = jnp.reshape(data[..., 1], (-1,))
        
        # Wrap phi to [-pi, pi] to ensure it falls within the histogram range
        # Assuming standard [-pi, pi] range for plotting
        phi_flat = (phi_flat + jnp.pi) % (2 * jnp.pi) - jnp.pi
        
        samples = jnp.stack([theta_flat, phi_flat], axis=-1)
        
        # Ranges: theta [0, pi], phi [-pi, pi]
        hist_range = [(0.0, jnp.pi), (-jnp.pi, jnp.pi)]
        
        # histogramdd returns (hist, edges), we only want the counts
        counts = jnp.histogramdd(samples, bins=self.bins, range=hist_range)[0]
        
        state["map"] += counts
        return {}, state

    def digest(self, all_values, state) -> dict[str, jnp.ndarray]:
        del all_values, state
        return {}


DEFAULT = Density2DEstimator
