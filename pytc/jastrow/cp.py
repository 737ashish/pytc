"""Canonical-polyadic polynomial pair Jastrow with upstream autodiff derivatives."""

import math
import operator
from typing import Optional

from flax import struct
import jax
import jax.numpy as jnp
from jax import random

from .jastrow import Jastrow


@struct.dataclass
class CP(Jastrow):
    """Per-nucleus CP networks defining the pair exponent ``u`` in ``J=exp(u)``.

    The polynomial residual is exchange-symmetrized and optionally augmented
    with ``0.5 * r12/(1+r12)`` outside the CP scale. Distances use
    ``sqrt(sum(displacement**2) + epsilon)``: positive epsilon smooths the
    coalescence point rather than imposing an exact cusp there. The cusp
    coefficient is spin-independent; no spin-resolved cusp is inferred.

    Construct with ``CP.create(mol, ...)``. For composition with a cusp-bearing
    Jastrow, use ``include_cusp=False`` to avoid adding the cusp twice.
    Coordinate/parameter derivatives and batches use the base Jastrow API.
    Nuclear positions are in the units returned by ``mol.atom_coords()``.

    Example::

        from pytc.jastrow import CP, BoysHandyAnalytical, CompositeJastrow

        cp = CP.create(mol, cp_degree=3, cp_rank=4)
        params = cp.init_params(key=jax.random.PRNGKey(0))
        u = cp._compute(r1, r2, params)
        grad_u, lap_u = cp.get_log_grads_r1(r1, r2, params)
        combined = CompositeJastrow.create([
            BoysHandyAnalytical.create(mol), CP.create(mol, include_cusp=False)
        ])
        combined_params = combined.init_params()
    """

    nuclear_pos: jax.Array
    cp_degree: int = struct.field(pytree_node=False, default=3)
    cp_rank: int = struct.field(pytree_node=False, default=4)
    cp_scale_init: float = struct.field(pytree_node=False, default=1.0)
    learnable_cp_scale: bool = struct.field(pytree_node=False, default=True)
    use_log_scale: bool = struct.field(pytree_node=False, default=False)
    use_cubic_features: bool = struct.field(pytree_node=False, default=False)
    use_manual_symmetrization: bool = struct.field(pytree_node=False, default=True)
    use_raw_distances: bool = struct.field(pytree_node=False, default=False)
    include_cusp: bool = struct.field(pytree_node=False, default=True)
    epsilon: float = struct.field(pytree_node=False, default=1e-8)
    name: Optional[str] = struct.field(pytree_node=False, default=None)

    @classmethod
    def create(cls, mol, cp_degree=3, cp_rank=4, cp_scale_init=1.0,
               learnable_cp_scale=True, use_log_scale=False,
               use_cubic_features=False, use_manual_symmetrization=True,
               use_raw_distances=False, include_cusp=True, epsilon=1e-8,
               name=None):
        """Create CP with fixed ``r/(1+r)`` scaling (or raw distances).

        ``cp_degree`` counts multiplicative polynomial stages; ``cp_rank``
        is their width. Cubic features supplement the squared distances.
        With manual symmetrization disabled, symmetric sums/products replace
        the separate electron-nucleus features. Parameters remain per nucleus,
        not shared by atomic species. A log scale must start strictly positive.
        """
        cp_degree, cp_rank = operator.index(cp_degree), operator.index(cp_rank)
        if cp_degree < 1 or cp_rank < 1:
            raise ValueError("cp_degree and cp_rank must be positive integers")
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and nonnegative")
        if not math.isfinite(cp_scale_init):
            raise ValueError("cp_scale_init must be finite")
        if learnable_cp_scale and use_log_scale and cp_scale_init <= 0:
            raise ValueError("A learnable log scale requires cp_scale_init > 0")
        if include_cusp and use_raw_distances:
            raise ValueError("include_cusp requires use_raw_distances=False")
        nuclear_pos = jnp.asarray(mol.atom_coords())
        if nuclear_pos.ndim != 2 or nuclear_pos.shape[1] != 3 or not len(nuclear_pos):
            raise ValueError("Expected at least one nuclear position with shape (3,)")
        return cls(
            nuclear_pos=nuclear_pos, cp_degree=cp_degree, cp_rank=cp_rank,
            cp_scale_init=float(cp_scale_init), learnable_cp_scale=learnable_cp_scale,
            use_log_scale=use_log_scale, use_cubic_features=use_cubic_features,
            use_manual_symmetrization=use_manual_symmetrization,
            use_raw_distances=use_raw_distances, include_cusp=include_cusp,
            epsilon=float(epsilon), name=name,
        )

    @property
    def natom(self):
        return self.nuclear_pos.shape[0]

    @property
    def cp_input_dim(self):
        return 6 if self.use_cubic_features else 3

    def init_params(self, **kwargs):
        """Initialize with ``key`` and optional ``cp_init_scale`` (default 0.01).

        Preserve the original initialization: factors at different polynomial
        stages start equal within each nucleus but are independently trainable.
        Fixed scale is model configuration, not an optimizer parameter.
        """
        key = kwargs.get("key", random.PRNGKey(0))
        amplitude = float(kwargs.get("cp_init_scale", 1e-2))
        key, *subkeys = random.split(key, self.natom + 1)
        factors = jnp.stack([
            amplitude * random.normal(k, (self.cp_input_dim, self.cp_rank))
            for k in subkeys
        ])
        key, subkey = random.split(key)
        params = {
            "cp_U": jnp.repeat(factors[:, None, :, :], self.cp_degree, axis=1),
            "cp_C": amplitude * random.normal(subkey, (self.natom, self.cp_rank)),
            "cp_bias": jnp.zeros(self.natom),
        }
        if self.learnable_cp_scale:
            if self.use_log_scale:
                params["cp_scale_log"] = jnp.log(jnp.asarray(self.cp_scale_init))
            else:
                params["cp_scale"] = jnp.asarray(self.cp_scale_init)
        return params

    def _get_cp_scale(self, params):
        if not self.learnable_cp_scale:
            return self.cp_scale_init
        if self.use_log_scale:
            return jnp.exp(params["cp_scale_log"])
        return params["cp_scale"]

    def _distance(self, displacement):
        r = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1) + self.epsilon)
        return r if self.use_raw_distances else r / (1.0 + r)

    def _features(self, a, b, c):
        powers = (2, 3) if self.use_cubic_features else (2,)
        if self.use_manual_symmetrization:
            return jnp.stack([v**p for p in powers for v in (a, b, c)])
        return jnp.stack([
            v for p in powers for v in (a**p + b**p, a**p * b**p, c**p)
        ])

    def _polynomial(self, x, factors, projection, bias):
        state = x @ factors[0]
        # Static degree: primitive operations also support folx differentiation.
        for stage in range(1, self.cp_degree):
            state = state * (x @ factors[stage]) + state
        return jnp.dot(state, projection) + bias

    def _compute(self, r1, r2, params):
        r1, r2 = jnp.reshape(r1, (3,)), jnp.reshape(r2, (3,))
        c = self._distance(r1 - r2)

        def nucleus_value(position, factors, projection, bias):
            a, b = self._distance(r1 - position), self._distance(r2 - position)
            value = self._polynomial(self._features(a, b, c), factors, projection, bias)
            if self.use_manual_symmetrization:
                other = self._polynomial(self._features(b, a, c), factors, projection, bias)
                value = 0.5 * (value + other)
            return value

        residual = jnp.sum(jax.vmap(nucleus_value)(
            self.nuclear_pos, params["cp_U"], params["cp_C"], params["cp_bias"]
        ))
        result = self._get_cp_scale(params) * residual
        return result + 0.5 * c if self.include_cusp else result

    def get_cp_param_count(self):
        """Number of trainable scalar parameters."""
        return (self.natom * (self.cp_degree * self.cp_input_dim * self.cp_rank
                              + self.cp_rank + 1) + int(self.learnable_cp_scale))
