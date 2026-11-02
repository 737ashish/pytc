"""Tests for the fixed-coefficient, regularized CP cusp."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto
from pytc.autodiff.jastrow import CP


class TestCPCusp(unittest.TestCase):
    def setUp(self):
        self.mol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g",
                         unit="bohr", verbose=0)
        self.key = jax.random.PRNGKey(42)
        # Keep the pair away from nuclei when probing coalescence.
        self.center = jnp.array([0.7, -0.4, 0.6])

    @staticmethod
    def cusp_value(r1, r2, epsilon):
        radius = jnp.sqrt(jnp.sum((r1-r2)**2) + epsilon)
        return 0.5 * radius / (1 + radius)

    def models(self, **kwargs):
        return (CP(self.mol, include_cusp=True, **kwargs),
                CP(self.mol, include_cusp=False, **kwargs))

    def test_cusp_validation(self):
        with self.assertRaisesRegex(ValueError, "BH-style scaled distances"):
            CP(self.mol, include_cusp=True, use_raw_distances=True)
        cp = CP(self.mol)
        self.assertTrue(cp.include_cusp)
        self.assertFalse(cp.use_raw_distances)

    def test_regularized_cusp_derivatives(self):
        cp, residual = self.models(cp_scale_init=2.3)
        params = cp.init_params(key=self.key)
        direction = jnp.array([1., 2., -2.]) / 3
        for distance in (0., 1e-5, 1e-4, 1e-3, 0.1):
            with self.subTest(distance=distance):
                r1 = self.center
                r2 = r1 + distance * direction
                delta = r1-r2
                radius = jnp.sqrt(jnp.sum(delta**2) + cp.epsilon)
                expected_grad = delta / (2*radius*(1+radius)**2)
                cusp = lambda x: self.cusp_value(x, r2, cp.epsilon)
                grad = cp.grad_r(r1,r2,params)-residual.grad_r(r1,r2,params)
                lapl = cp.laplacian_r(r1,r2,params)-residual.laplacian_r(r1,r2,params)
                value = cp._compute(r1,r2,params)-residual._compute(r1,r2,params)
                np.testing.assert_allclose(value, cusp(r1), rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(grad, expected_grad, rtol=1e-9, atol=1e-11)
                np.testing.assert_allclose(grad, jax.grad(cusp)(r1), rtol=1e-9, atol=1e-11)
                np.testing.assert_allclose(lapl, jnp.trace(jax.hessian(cusp)(r1)),
                                           rtol=1e-9, atol=1e-9)

    def test_cusp_limit_outside_smoothing_region(self):
        # First reduce epsilon at a fixed small distance, approaching the
        # unsmoothed radial derivative 0.5/(1+r)^2, not exactly 0.5.
        distance = 1e-3
        target = 0.5/(1+distance)**2
        errors = []
        for epsilon in (1e-6, 1e-8, 1e-10, 1e-12):
            cp, residual = self.models(epsilon=epsilon)
            params = cp.init_params(key=self.key)
            r1 = self.center
            r2 = r1 + jnp.array([distance, 0., 0.])
            radial = -(cp.grad_r(r1,r2,params)-residual.grad_r(r1,r2,params))[0]
            errors.append(abs(float(radial)-target))
        self.assertTrue(all(b < a for a,b in zip(errors,errors[1:])))
        self.assertLess(errors[-1], 1e-6)

        # Then approach coalescence while sqrt(epsilon)/r also tends to zero.
        errors = []
        for distance in (1e-2, 1e-3, 1e-4):
            cp, residual = self.models(epsilon=distance**4)
            params = cp.init_params(key=self.key)
            r1 = self.center
            r2 = r1 + jnp.array([distance, 0., 0.])
            radial = -(cp.grad_r(r1,r2,params)-residual.grad_r(r1,r2,params))[0]
            errors.append(abs(float(radial)-0.5))
        self.assertTrue(all(b < a for a,b in zip(errors,errors[1:])))
        self.assertLess(errors[-1], 1.1e-4)

    def test_cusp_independent_of_cp_scale(self):
        r1, r2 = self.center, self.center+jnp.array([0.2,-0.1,0.3])
        expected = self.cusp_value(r1,r2,1e-8)
        for mode in ("direct", "log", "fixed"):
            for scale in ((0.,0.3,2.5) if mode != "log" else (0.3,1.,2.5)):
                with self.subTest(mode=mode, scale=scale):
                    cp,residual = self.models(cp_scale_init=scale,
                        learnable_cp_scale=mode!="fixed", use_log_scale=mode=="log")
                    params = cp.init_params(key=self.key)
                    difference = lambda p: cp._compute(r1,r2,p)-residual._compute(r1,r2,p)
                    np.testing.assert_allclose(difference(params), expected, rtol=1e-10, atol=1e-12)
                    # The cusp contributes no derivative to any learned parameter.
                    for leaf in jax.tree_util.tree_leaves(jax.grad(difference)(params)):
                        np.testing.assert_allclose(leaf, 0., atol=1e-12)
                    # The residual itself still scales, including when cusp is off.
                    unit = dict(params)
                    key = "cp_scale_log" if mode=="log" else ("_cp_scale_fixed" if mode=="fixed" else "cp_scale")
                    unit[key] = jnp.array(0. if mode=="log" else 1.)
                    np.testing.assert_allclose(residual._compute(r1,r2,params),
                        scale*residual._compute(r1,r2,unit), rtol=1e-10, atol=1e-12)

    def test_full_derivatives_at_nonunit_scale(self):
        r1, r2 = self.center, self.center+jnp.array([0.2,-0.1,0.3])
        for mode in ("direct", "log", "fixed"):
            cp = CP(self.mol, cp_scale_init=2.5,
                    learnable_cp_scale=mode!="fixed", use_log_scale=mode=="log")
            params = cp.init_params(key=self.key)
            for arg, getter in ((0,cp.get_log_grads_r1),(1,cp.get_log_grads_r2)):
                with self.subTest(mode=mode, electron=arg+1):
                    value = lambda x,y: cp._compute(x,y,params)
                    grad, lapl = getter(r1,r2,params)
                    np.testing.assert_allclose(grad, jax.grad(value,argnums=arg)(r1,r2),
                                               rtol=1e-8, atol=1e-10)
                    np.testing.assert_allclose(lapl, jnp.trace(jax.hessian(value,argnums=arg)(r1,r2)),
                                               rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
