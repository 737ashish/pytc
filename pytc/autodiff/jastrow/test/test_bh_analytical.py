"""Test analytical derivatives for Boys-Handy Jastrow factor."""

import unittest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import numpy as np

from pyscf import gto
from pytc.autodiff.jastrow import BoysHandy


class TestBoysHandyAnalytical(unittest.TestCase):
    """Test analytical gradient implementation against autodiff."""
    
    def setUp(self):
        """Set up test molecule and Jastrow factor."""
        # Create a simple molecule (Be atom)
        self.mol = gto.M(atom='Be 0 0 0', basis='cc-pvdz', unit='bohr')
        self.bh = BoysHandy(self.mol)
        self.params = self.bh.init_params()
        
        # Create random electron positions
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        self.r1 = random.normal(key1, (3,)) * 0.5
        self.r2 = random.normal(key2, (3,)) * 0.5
    
    def test_grad_r1_vs_autodiff(self):
        """Test analytical gradient w.r.t r1 matches autodiff."""
        # Compute analytical gradient
        grad_analytical = self.bh._grad_r1_analytical(self.r1, self.r2, self.params)
        
        # Compute autodiff gradient using base class method
        def scalar_fn(x):
            return self.bh._compute(x, self.r2, self.params).reshape(-1)[0]
        grad_autodiff = jax.grad(scalar_fn)(self.r1)
        
        # Check they match
        np.testing.assert_allclose(
            grad_analytical, grad_autodiff, 
            rtol=1e-6, atol=1e-8,
            err_msg="Analytical gradient w.r.t r1 does not match autodiff"
        )
        print("✓ Analytical gradient w.r.t r1 matches autodiff")
    
    def test_grad_r2_vs_autodiff(self):
        """Test analytical gradient w.r.t r2 matches autodiff."""
        # Compute analytical gradient
        grad_analytical = self.bh._grad_r2_analytical(self.r1, self.r2, self.params)
        
        # Compute autodiff gradient using base class method
        def scalar_fn(x):
            return self.bh._compute(self.r1, x, self.params).reshape(-1)[0]
        grad_autodiff = jax.grad(scalar_fn)(self.r2)
        
        # Check they match
        np.testing.assert_allclose(
            grad_analytical, grad_autodiff, 
            rtol=1e-6, atol=1e-8,
            err_msg="Analytical gradient w.r.t r2 does not match autodiff"
        )
        print("✓ Analytical gradient w.r.t r2 matches autodiff")
    
    def test_grad_r_override(self):
        """Test that grad_r method uses analytical implementation."""
        # Call overridden method
        grad_override = self.bh.grad_r(self.r1, self.r2, self.params)
        
        # Call analytical method directly
        grad_analytical = self.bh._grad_r1_analytical(self.r1, self.r2, self.params)
        
        # Should be identical
        np.testing.assert_array_equal(
            grad_override, grad_analytical,
            err_msg="grad_r override does not call analytical implementation"
        )
        print("✓ grad_r method correctly uses analytical implementation")
    
    def test_get_log_grads_r1(self):
        """Test get_log_grads_r1 returns analytical gradient."""
        grad_u, lapl_u = self.bh.get_log_grads_r1(self.r1, self.r2, self.params)
        
        # Check gradient matches analytical
        grad_analytical = self.bh._grad_r1_analytical(self.r1, self.r2, self.params)
        np.testing.assert_allclose(
            grad_u, grad_analytical,
            rtol=1e-10, atol=1e-12,
            err_msg="get_log_grads_r1 gradient does not match analytical"
        )
        
        # Check Laplacian is scalar
        self.assertEqual(lapl_u.shape, ())
        print("✓ get_log_grads_r1 returns analytical gradient and scalar Laplacian")
    
    def test_get_log_grads_r2(self):
        """Test get_log_grads_r2 returns analytical gradient."""
        grad_u, lapl_u = self.bh.get_log_grads_r2(self.r1, self.r2, self.params)
        
        # Check gradient matches analytical
        grad_analytical = self.bh._grad_r2_analytical(self.r1, self.r2, self.params)
        np.testing.assert_allclose(
            grad_u, grad_analytical,
            rtol=1e-10, atol=1e-12,
            err_msg="get_log_grads_r2 gradient does not match analytical"
        )
        
        # Check Laplacian is scalar
        self.assertEqual(lapl_u.shape, ())
        print("✓ get_log_grads_r2 returns analytical gradient and scalar Laplacian")
    
    def test_multiple_positions(self):
        """Test analytical gradients work for multiple random positions."""
        key = random.PRNGKey(123)

        for i in range(5):
            key, key1, key2 = random.split(key, 3)
            r1 = random.normal(key1, (3,)) * 1.0
            r2 = random.normal(key2, (3,)) * 1.0

            # Test r1 gradient
            grad_analytical = self.bh._grad_r1_analytical(r1, r2, self.params)
            def scalar_fn(x):
                return self.bh._compute(x, r2, self.params).reshape(-1)[0]
            grad_autodiff = jax.grad(scalar_fn)(r1)

            np.testing.assert_allclose(
                grad_analytical, grad_autodiff,
                rtol=1e-6, atol=1e-8,
                err_msg=f"Mismatch at position {i}"
            )

        print("✓ Analytical gradients match autodiff for multiple random positions")

    def test_laplacian_r1_vs_autodiff(self):
        """Test analytical Laplacian w.r.t r1 matches autodiff."""
        # Compute analytical Laplacian
        lapl_analytical = self.bh._laplacian_r1_analytical(self.r1, self.r2, self.params)

        # Compute autodiff Laplacian
        def scalar_fn(x):
            return self.bh._compute(x, self.r2, self.params).reshape(-1)[0]
        lapl_autodiff = jnp.trace(jax.hessian(scalar_fn)(self.r1))

        # Check they match
        np.testing.assert_allclose(
            lapl_analytical, lapl_autodiff,
            rtol=1e-5, atol=1e-7,
            err_msg="Analytical Laplacian w.r.t r1 does not match autodiff"
        )
        print("✓ Analytical Laplacian w.r.t r1 matches autodiff")

    def test_laplacian_r2_vs_autodiff(self):
        """Test analytical Laplacian w.r.t r2 matches autodiff."""
        # Compute analytical Laplacian
        lapl_analytical = self.bh._laplacian_r2_analytical(self.r1, self.r2, self.params)

        # Compute autodiff Laplacian
        def scalar_fn(x):
            return self.bh._compute(self.r1, x, self.params).reshape(-1)[0]
        lapl_autodiff = jnp.trace(jax.hessian(scalar_fn)(self.r2))

        # Check they match
        np.testing.assert_allclose(
            lapl_analytical, lapl_autodiff,
            rtol=1e-5, atol=1e-7,
            err_msg="Analytical Laplacian w.r.t r2 does not match autodiff"
        )
        print("✓ Analytical Laplacian w.r.t r2 matches autodiff")

    def test_laplacian_r_override(self):
        """Test that laplacian_r method uses analytical implementation."""
        # Call overridden method
        lapl_override = self.bh.laplacian_r(self.r1, self.r2, self.params)

        # Call analytical method directly
        lapl_analytical = self.bh._laplacian_r1_analytical(self.r1, self.r2, self.params)

        # Should be identical
        np.testing.assert_allclose(
            lapl_override, lapl_analytical,
            rtol=1e-12, atol=1e-14,
            err_msg="laplacian_r override does not call analytical implementation"
        )
        print("✓ laplacian_r method correctly uses analytical implementation")

    def test_get_log_grads_uses_analytical_laplacian(self):
        """Test that get_log_grads methods use analytical Laplacian."""
        # Test r1
        grad_u_r1, lapl_u_r1 = self.bh.get_log_grads_r1(self.r1, self.r2, self.params)
        lapl_analytical_r1 = self.bh._laplacian_r1_analytical(self.r1, self.r2, self.params)

        np.testing.assert_allclose(
            lapl_u_r1, lapl_analytical_r1,
            rtol=1e-12, atol=1e-14,
            err_msg="get_log_grads_r1 Laplacian does not match analytical"
        )

        # Test r2
        grad_u_r2, lapl_u_r2 = self.bh.get_log_grads_r2(self.r1, self.r2, self.params)
        lapl_analytical_r2 = self.bh._laplacian_r2_analytical(self.r1, self.r2, self.params)

        np.testing.assert_allclose(
            lapl_u_r2, lapl_analytical_r2,
            rtol=1e-12, atol=1e-14,
            err_msg="get_log_grads_r2 Laplacian does not match analytical"
        )

        print("✓ get_log_grads methods use analytical Laplacian")


if __name__ == '__main__':
    unittest.main()

