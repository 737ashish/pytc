"""Analytical CP parity and regressions; run with unittest discovery."""

import itertools
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto

from pytc.jastrow import CP, CPAnalytical, BoysHandyAnalytical, CompositeJastrow
from pytc.vmc.hamiltonian import compute_jastrow_terms

jax.config.update("jax_enable_x64", True)
R1 = jnp.array([0.3, -0.2, 0.6])
R2 = jnp.array([-0.5, 0.4, -0.3])


def parameters(model):
    p = model.init_params(key=jax.random.PRNGKey(17), cp_init_scale=0.2)
    p["cp_U"] = 0.2 * jax.random.normal(jax.random.PRNGKey(31), p["cp_U"].shape)
    p["cp_bias"] = jnp.arange(model.natom) * 0.03
    return p


def assert_tree_close(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        assert np.isfinite(x).all() and np.isfinite(y).all()
        np.testing.assert_allclose(x, y, rtol=2e-8, atol=2e-9)


class TestCPAnalytical(unittest.TestCase):
    def test_distance_laplacian_with_smoothing(self):
        for raw, epsilon in itertools.product((False, True), (0.0, 1e-16, 1e-8, 0.1)):
            model = CPAnalytical.create(self.mol, use_raw_distances=raw,
                                        include_cusp=not raw, epsilon=epsilon)
            points = [jnp.array([0.2, -0.1, 0.5]), jnp.array([1e-6, 0., 0.])]
            if epsilon:
                points.append(jnp.zeros(3))
            for point in points:
                with self.subTest(raw=raw, epsilon=epsilon, point=point):
                    value, gradient, laplacian = model._distance_derivatives(point)
                    assert_tree_close(value, model._distance(point))
                    assert_tree_close(gradient, jax.grad(model._distance)(point))
                    assert_tree_close(laplacian, jnp.trace(jax.hessian(model._distance)(point)))

    def test_polynomial_laplacian_at_zero_multiplier(self):
        model = CPAnalytical.create(self.mol, cp_degree=3)
        x = jnp.array([1., 0., 0.])
        # Each rank channel is x0*(1-x0); at x0=1 its value is zero
        # but its gradient and Laplacian are nonzero.
        factors = jnp.zeros((3, 3, 4)).at[0, 0, :].set(1.).at[1, 0, :].set(-1.)
        projection = jnp.ones(4)
        gradient, laplacian = model._polynomial_derivatives(
            x, jnp.eye(3), jnp.zeros(3), factors, projection
        )
        np.testing.assert_array_equal(gradient, jnp.array([-4., 0., 0.]))
        np.testing.assert_array_equal(laplacian, -8.)

    def test_higher_degree_random_points_and_cusp_off(self):
        for manual, epsilon in itertools.product((False, True), (0.0, 0.03)):
            options = dict(cp_degree=5, cp_rank=7, use_cubic_features=True,
                           use_manual_symmetrization=manual, include_cusp=False,
                           epsilon=epsilon)
            cp, cpa = CP.create(self.mol, **options), CPAnalytical.create(self.mol, **options)
            p = parameters(cp)
            points = jax.random.normal(jax.random.PRNGKey(81), (3, 2, 3))
            for r1, r2 in points:
                with self.subTest(manual=manual, epsilon=epsilon, r1=r1):
                    f = lambda x: cp._compute(x, r2, p)
                    expected = jax.grad(f)(r1), jnp.trace(jax.hessian(f)(r1))
                    assert_tree_close(cpa.get_log_grads_r1(r1, r2, p), expected)

    def test_zero_scale_retains_cusp_derivatives(self):
        model = CPAnalytical.create(self.mol)
        p = {**parameters(model), "cp_scale": jnp.array(0.)}
        expected = model._distance_derivatives(R1-R2)
        assert_tree_close(model.get_log_grads_r1(R1, R2, p),
                          (0.5*expected[1], 0.5*expected[2]))

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g", verbose=0)

    def test_factory_forward_and_initialization(self):
        cp, cpa = CP.create(self.mol), CPAnalytical.create(self.mol)
        self.assertIs(type(cp), CP)
        self.assertIs(type(cpa), CPAnalytical)
        self.assertIs(CPAnalytical._compute, CP._compute)
        self.assertIs(CPAnalytical.init_params, CP.init_params)
        assert_tree_close(cp.init_params(), cpa.init_params())
        p = parameters(cp)
        self.assertEqual(cp.get_cp_param_count(), cpa.get_cp_param_count())
        np.testing.assert_array_equal(cp._compute(R1, R2, p), cpa._compute(R1, R2, p))
        assert_tree_close(cp.grad_params(R1, R2, p), cpa.grad_params(R1, R2, p))
        fn = jax.jit(lambda model, params: model.get_log_grads_r1(R1, R2, params))
        assert_tree_close(fn(cp, p), fn(cpa, p))

    def test_all_feature_modes(self):
        for degree, cubic, manual, raw in itertools.product((1, 3), (False, True), (False, True), (False, True)):
            with self.subTest(degree=degree, cubic=cubic, manual=manual, raw=raw):
                opts = dict(cp_degree=degree, use_cubic_features=cubic,
                            use_manual_symmetrization=manual, use_raw_distances=raw,
                            include_cusp=not raw)
                cp, cpa = CP.create(self.mol, **opts), CPAnalytical.create(self.mol, **opts)
                p = parameters(cp)
                assert_tree_close(cp.get_log_grads_r1(R1, R2, p), cpa.get_log_grads_r1(R1, R2, p))
                assert_tree_close(cp.get_log_grads_r2(R1, R2, p), cpa.get_log_grads_r2(R1, R2, p))
                assert_tree_close(cp.grad_r(R1, R2, p), cpa.grad_r(R1, R2, p))
                assert_tree_close(cp.laplacian_r(R1, R2, p), cpa.laplacian_r(R1, R2, p))

    def test_symmetric_product_chain_rule(self):
        # Isolate x=a^p*b^p: the archived helper doubled/tripled its derivative.
        for power in (2, 3):
            with self.subTest(power=power):
                model = CPAnalytical.create(self.mol, use_manual_symmetrization=False,
                                            use_cubic_features=power == 3)
                derivatives = model._feature_gradients(
                    1.0, 1.0, 1.0, jnp.array([1., 0., 0.]), jnp.zeros(3), jnp.zeros(3)
                )
                index = 1 if power == 2 else 4
                np.testing.assert_array_equal(derivatives[index], jnp.array([power, 0., 0.]))

    def test_zero_polynomial_factors(self):
        model = CPAnalytical.create(self.mol, cp_degree=3)
        x = jnp.array([1., 0., 0.])
        factors = jnp.zeros((3, 3, 4)).at[1, 0, :].set(-1.0)
        projection = jnp.ones(4)
        dx = jnp.eye(3)
        expected = jax.grad(lambda t: model._polynomial(t, factors, projection, 0.))(x)
        assert_tree_close(model._polynomial_gradient(x, dx, factors, projection), expected)

    def test_scale_and_parameter_gradients_through_derivatives(self):
        for mode, manual in itertools.product(("direct", "log", "fixed"), (False, True)):
            with self.subTest(mode=mode, manual=manual):
                opts = dict(cp_scale_init=0.7, use_log_scale=mode == "log",
                            learnable_cp_scale=mode != "fixed", use_cubic_features=True,
                            use_manual_symmetrization=manual)
                cp, cpa = CP.create(self.mol, **opts), CPAnalytical.create(self.mol, **opts)
                p = parameters(cp)
                def objective(model, params):
                    g, lap = model.get_log_grads_r1(R1, R2, params)
                    return jnp.sum(g*g) + lap
                grad = jax.jit(jax.grad(objective, argnums=1))
                assert_tree_close(grad(cp, p), grad(cpa, p))
                zero = {**p, "cp_C": jnp.zeros_like(p["cp_C"]), "cp_bias": jnp.zeros_like(p["cp_bias"])}
                assert_tree_close(cp.get_log_grads_r1(R1, R2, zero), cpa.get_log_grads_r1(R1, R2, zero))

    def test_smoothed_coalescence(self):
        for raw, manual in itertools.product((False, True), (False, True)):
            opts = dict(use_raw_distances=raw, include_cusp=not raw,
                        use_manual_symmetrization=manual, use_cubic_features=True)
            cp, cpa = CP.create(self.mol, **opts), CPAnalytical.create(self.mol, **opts)
            p = parameters(cp)
            for r1, r2 in ((R1, R1), (cp.nuclear_pos[0], R2), (R1+1e-6, R1)):
                with self.subTest(raw=raw, manual=manual, r1=r1, r2=r2):
                    assert_tree_close(cp.get_log_grads_r1(r1, r2, p), cpa.get_log_grads_r1(r1, r2, p))

    def test_batches_and_vmc_composition(self):
        cp, cpa = CP.create(self.mol, include_cusp=False), CPAnalytical.create(self.mol, include_cusp=False)
        p = parameters(cp)
        coords = jnp.stack([R1, R2, R1+0.4])
        assert_tree_close(jax.jit(cp.grad_r_batch)(coords, coords[:2], p),
                          jax.jit(cpa.grad_r_batch)(coords, coords[:2], p))
        assert_tree_close(jax.jit(cp.get_pair_grid_grad_lap)(coords, p),
                          jax.jit(cpa.get_pair_grid_grad_lap)(coords, p))
        bh = BoysHandyAnalytical.create(self.mol)
        for components in ([cp], [bh, cp]):
            models = components[:-1] + [cpa]
            left = CompositeJastrow.create(components)
            right = CompositeJastrow.create(models)
            params = [bh.init_params(), p] if len(models) == 2 else [p]
            assert_tree_close(compute_jastrow_terms(SimpleNamespace(jastrow=left), coords, params),
                              compute_jastrow_terms(SimpleNamespace(jastrow=right), coords, params))

    def test_finite_difference_multinucleus(self):
        for n in (1, 3):
            geom = SimpleNamespace(atom_coords=lambda: np.array([[0., 0., 1.4*i] for i in range(n)]))
            model = CPAnalytical.create(geom, use_cubic_features=True, use_manual_symmetrization=False)
            p = parameters(model)
            h = 1e-4
            center = model._compute(R1, R2, p)
            plus = jnp.array([model._compute(R1+d, R2, p) for d in h*jnp.eye(3)])
            minus = jnp.array([model._compute(R1-d, R2, p) for d in h*jnp.eye(3)])
            g, lap = model.get_log_grads_r1(R1, R2, p)
            np.testing.assert_allclose(g, (plus-minus)/(2*h), rtol=2e-6, atol=1e-8)
            np.testing.assert_allclose(lap, jnp.sum(plus-2*center+minus)/h**2, rtol=2e-5, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
