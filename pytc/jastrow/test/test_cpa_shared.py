"""Shared CPA pair grids against the generic per-pair reference."""
import itertools
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from pytc.jastrow import CPAnalytical, BoysHandyAnalytical, CompositeJastrow
from pytc.vmc.hamiltonian import compute_jastrow_terms
from pyscf import gto

jax.config.update("jax_enable_x64", True)


def close(a, b, rtol=2e-8, atol=2e-9):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        assert np.isfinite(x).all() and np.isfinite(y).all()
        np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)


class TestSharedCPA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom="O 0 0 0; H 0 -1.4 1.1; H 0 1.4 1.1", unit="bohr", basis="sto-3g", verbose=0)
        cls.coords = 0.5*jax.random.normal(jax.random.PRNGKey(81), (4, 3))

    def models(self, **options):
        generic = CPAnalytical.create(self.mol, use_shared_pair_grid=False, **options)
        shared = generic.replace(use_shared_pair_grid=True)
        p = generic.init_params(key=jax.random.PRNGKey(7), cp_init_scale=0.2)
        p["cp_U"] = 0.2*jax.random.normal(jax.random.PRNGKey(31), p["cp_U"].shape)
        return generic, shared, p

    def test_all_feature_modes(self):
        for degree, cubic, manual, raw in itertools.product((1, 3), (False, True), (False, True), (False, True)):
            with self.subTest(degree=degree, cubic=cubic, manual=manual, raw=raw):
                generic, shared, p = self.models(cp_degree=degree, use_cubic_features=cubic,
                    use_manual_symmetrization=manual, use_raw_distances=raw, include_cusp=not raw)
                close(jax.jit(shared.get_pair_grid_grad_lap)(self.coords, p),
                      jax.jit(generic.get_pair_grid_grad_lap)(self.coords, p))
                close(jax.jit(shared.grad_r_batch)(self.coords, self.coords[:2]+0.1, p),
                      jax.jit(generic.grad_r_batch)(self.coords, self.coords[:2]+0.1, p))

    def test_scales_and_single_point(self):
        for mode in ("direct", "log", "fixed"):
            generic, shared, p = self.models(cp_scale_init=0.7, learnable_cp_scale=mode!="fixed",
                                             use_log_scale=mode=="log", use_cubic_features=True)
            x = self.coords[:1]
            close(shared.get_pair_grid_grad_lap(x, p), generic.get_pair_grid_grad_lap(x, p))
            close(shared.grad_r_batch(x, x+0.1, p), generic.grad_r_batch(x, x+0.1, p))
        p = {**p, "cp_bias": jnp.full_like(p["cp_bias"], 100.)}
        close(shared.get_pair_grid_grad_lap(x, p), generic.get_pair_grid_grad_lap(x, p))

    def test_diagonal_and_distinct_coincident_electrons(self):
        generic, shared, p = self.models(epsilon=1e-8)
        coords = self.coords.at[1].set(self.coords[0])
        g, lap = shared.get_pair_grid_grad_lap(coords, p)
        close((g, lap), generic.get_pair_grid_grad_lap(coords, p))
        close((g[0, 1], lap[0, 1]), shared.get_log_grads_r1(coords[0], coords[1], p))
        close((g[0, 0], lap[0, 0]), shared.get_log_grads_r1(coords[0], coords[0]+1., p))
        # Rectangular batches must NOT displace even their index diagonal.
        close(shared.grad_r_batch(coords, coords, p), generic.grad_r_batch(coords, coords, p))

    def test_zero_smoothing_safe_diagonal(self):
        generic, shared, p = self.models(epsilon=0.)
        close(shared.get_pair_grid_grad_lap(self.coords, p), generic.get_pair_grid_grad_lap(self.coords, p))

    def test_parameter_gradients(self):
        for manual in (False, True):
            generic, shared, p = self.models(use_cubic_features=True, use_manual_symmetrization=manual,
                                             use_log_scale=True)
            def pair_objective(model, params):
                g, lap = model.get_pair_grid_grad_lap(self.coords, params)
                mask = 1-jnp.eye(len(self.coords))
                g = jnp.sum(g*mask[..., None], axis=1)
                return jnp.sum(lap*mask) + jnp.sum(g*g)
            def rectangular_objective(model, params):
                g = model.grad_r_batch(self.coords, self.coords[:2]+0.2, params)
                return jnp.sum(g*g)
            for objective in (pair_objective, rectangular_objective):
                grad = jax.jit(jax.grad(objective, argnums=1))
                close(grad(shared, p), grad(generic, p))

    def test_coordinate_gradients_through_cache(self):
        generic, shared, p = self.models(use_cubic_features=True)
        def objective(model, x):
            g, lap = model.get_pair_grid_grad_lap(x, p)
            return jnp.sum(lap) + jnp.sum(g*g)
        grad = jax.jit(jax.grad(objective, argnums=1))
        close(grad(shared, self.coords), grad(generic, self.coords))

    def test_composite_vmc(self):
        generic, shared, p = self.models(include_cusp=False)
        bh = BoysHandyAnalytical.create(self.mol)
        params = [bh.init_params(), p]
        left = SimpleNamespace(jastrow=CompositeJastrow.create([bh, generic]))
        right = SimpleNamespace(jastrow=CompositeJastrow.create([bh, shared]))
        close(compute_jastrow_terms(left, self.coords, params), compute_jastrow_terms(right, self.coords, params))

    def test_single_nucleus_and_higher_rank(self):
        geom = SimpleNamespace(atom_coords=lambda: np.array([[0., 0., 0.]]))
        generic = CPAnalytical.create(geom, cp_degree=5, cp_rank=7, use_cubic_features=True)
        shared = generic.replace(use_shared_pair_grid=True)
        p = generic.init_params(cp_init_scale=0.2)
        close(shared.get_pair_grid_grad_lap(self.coords, p), generic.get_pair_grid_grad_lap(self.coords, p))

    def test_float32(self):
        generic, shared, p = self.models()
        generic = generic.replace(nuclear_pos=generic.nuclear_pos.astype(jnp.float32))
        shared = shared.replace(nuclear_pos=shared.nuclear_pos.astype(jnp.float32))
        p = jax.tree.map(lambda x: x.astype(jnp.float32), p)
        x = self.coords.astype(jnp.float32)
        a = jax.jit(shared.get_pair_grid_grad_lap)(x, p)
        b = jax.jit(generic.get_pair_grid_grad_lap)(x, p)
        self.assertEqual(a[0].dtype, jnp.float32)
        close(a, b, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
