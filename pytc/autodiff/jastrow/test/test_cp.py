"""
Test suite for standalone CP Jastrow factor.
"""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from pyscf import gto

from pytc.autodiff.jastrow import CP, BoysHandy, CompositeJastrow


class TestCP(unittest.TestCase):
    """Test cases for standalone CP Jastrow factor."""
    
    def setUp(self):
        """Set up test molecule and parameters."""
        # Create H2 molecule
        self.mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='bohr', verbose=0)
        
        # Test positions
        self.r1 = jnp.array([0.5, 0.0, 0.0])
        self.r2 = jnp.array([-0.5, 0.0, 0.0])
        
        # Random key for parameter initialization
        self.key = random.PRNGKey(42)
    
    def test_initialization(self):
        """Test CP initialization with different configurations."""
        # Default configuration
        cp = CP(self.mol)
        self.assertEqual(cp.cp_degree, 3)
        self.assertEqual(cp.cp_rank, 4)
        self.assertFalse(cp.use_raw_distances)
        self.assertTrue(cp.include_cusp)
        self.assertTrue(cp.use_manual_symmetrization)
        
        # Raw-distance CP is available explicitly without a cusp term.
        cp_raw = CP(self.mol, use_raw_distances=True, include_cusp=False)
        self.assertTrue(cp_raw.use_raw_distances)
        self.assertFalse(cp_raw.include_cusp)
        
        # Custom configuration
        cp_custom = CP(self.mol, cp_degree=5, cp_rank=8, use_cubic_features=True)
        self.assertEqual(cp_custom.cp_degree, 5)
        self.assertEqual(cp_custom.cp_rank, 8)
        self.assertEqual(cp_custom.cp_input_dim, 6)  # With cubic features
    
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        cp = CP(self.mol, cp_degree=3, cp_rank=4)
        params = cp.init_params(key=self.key)
        
        # Check parameter shapes
        self.assertEqual(params['cp_U'].shape, (2, 3, 3, 4))  # (natom, degree, input_dim, rank)
        self.assertEqual(params['cp_C'].shape, (2, 4))  # (natom, rank)
        self.assertEqual(params['cp_bias'].shape, (2,))  # (natom,)
        
        # Check scale parameter
        self.assertIn('cp_scale', params)
        
        # Test with log scale
        cp_log = CP(self.mol, use_log_scale=True)
        params_log = cp_log.init_params(key=self.key)
        self.assertIn('cp_scale_log', params_log)
        self.assertNotIn('cp_scale', params_log)
    
    def test_compute(self):
        """Test CP computation."""
        cp = CP(self.mol, cp_degree=3, cp_rank=4)
        params = cp.init_params(key=self.key)
        
        # Compute Jastrow exponent
        u = cp._compute(self.r1, self.r2, params)
        
        # Should be a scalar
        self.assertEqual(u.shape, ())
        
        # Should be finite
        self.assertTrue(jnp.isfinite(u))
    
    def test_electron_exchange_symmetry(self):
        """Test that u(r1, r2) = u(r2, r1)."""
        cp = CP(self.mol, cp_degree=3, cp_rank=4)
        params = cp.init_params(key=self.key)
        
        u_12 = cp._compute(self.r1, self.r2, params)
        u_21 = cp._compute(self.r2, self.r1, params)
        
        # Should be symmetric
        self.assertAlmostEqual(float(u_12), float(u_21), places=10)
    
    def test_gradient(self):
        """Test gradient computation."""
        cp = CP(self.mol, cp_degree=3, cp_rank=4)
        params = cp.init_params(key=self.key)
        
        # Compute gradient
        grad = cp.grad_r(self.r1, self.r2, params)
        
        # Should be shape (3,)
        self.assertEqual(grad.shape, (3,))
        
        # Should be finite
        self.assertTrue(jnp.all(jnp.isfinite(grad)))
        
        # Compare with autodiff
        def u_fn(r):
            return cp._compute(r, self.r2, params)
        
        grad_autodiff = jax.grad(u_fn)(self.r1)

        # Should match (within numerical precision)
        np.testing.assert_allclose(grad, grad_autodiff, rtol=1e-5, atol=1e-8)
    
    def test_laplacian(self):
        """Test Laplacian computation."""
        cp = CP(self.mol, cp_degree=3, cp_rank=4)
        params = cp.init_params(key=self.key)
        
        # Compute Laplacian
        lapl = cp.laplacian_r(self.r1, self.r2, params)
        
        # Should be a scalar
        self.assertEqual(lapl.shape, ())
        
        # Should be finite
        self.assertTrue(jnp.isfinite(lapl))
    
    def test_get_log_grads(self):
        """Test get_log_grads methods."""
        cp = CP(self.mol, cp_degree=3, cp_rank=4)
        params = cp.init_params(key=self.key)
        
        # Test r1
        grad_r1, lapl_r1 = cp.get_log_grads_r1(self.r1, self.r2, params)
        self.assertEqual(grad_r1.shape, (3,))
        self.assertEqual(lapl_r1.shape, ())
        
        # Test r2
        grad_r2, lapl_r2 = cp.get_log_grads_r2(self.r1, self.r2, params)
        self.assertEqual(grad_r2.shape, (3,))
        self.assertEqual(lapl_r2.shape, ())
    
    def test_composite_with_bh(self):
        """Test CP + BoysHandy composition (equivalent to CPBH)."""
        bh = BoysHandy(self.mol)
        # BH supplies the cusp; CP supplies only the polynomial residual.
        cp = CP(self.mol, cp_degree=3, cp_rank=4, use_raw_distances=False,
                include_cusp=False)
        composite = CompositeJastrow([bh, cp])
        
        # Initialize parameters
        params_composite = composite.init_params()
        
        # Should have 2 parameter sets
        self.assertEqual(len(params_composite), 2)
        
        # Compute value
        u = composite._compute(self.r1, self.r2, params_composite)
        self.assertTrue(jnp.isfinite(u))
    


if __name__ == '__main__':
    unittest.main()

