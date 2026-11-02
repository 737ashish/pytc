"""
Test that CompositeJastrow([BoysHandy, CP]) gives same results as CPBH.
"""
import unittest
import numpy as np
import jax.numpy as jnp
from jax import random
from pyscf import gto

from pytc.autodiff.jastrow import CP, BoysHandy, CPBH, CompositeJastrow


class TestCPvsCPBH(unittest.TestCase):
    """Test that CP + BH composition matches CPBH."""
    
    def setUp(self):
        """Set up test molecule and parameters."""
        # Create H2 molecule
        self.mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='bohr', verbose=0)
        
        # Test positions
        self.r1 = jnp.array([0.5, 0.0, 0.0])
        self.r2 = jnp.array([-0.5, 0.0, 0.0])
        
        # Random key for parameter initialization
        self.key = random.PRNGKey(42)
        
        # CP configuration
        self.cp_degree = 3
        self.cp_rank = 4
        self.cp_scale_init = 1.0
    
    def test_same_value_with_same_params(self):
        """Test that u(r1, r2) is the same for CPBH and CompositeJastrow([BH, CP])."""
        # Create CPBH
        cpbh = CPBH(self.mol, cp_degree=self.cp_degree, cp_rank=self.cp_rank, 
                    cp_scale_init=self.cp_scale_init)
        params_cpbh = cpbh.init_params(key=self.key)
        
        # BH supplies the cusp; CP supplies only the polynomial residual.
        bh = BoysHandy(self.mol)
        cp = CP(self.mol, cp_degree=self.cp_degree, cp_rank=self.cp_rank,
                cp_scale_init=self.cp_scale_init, use_raw_distances=False,
                include_cusp=False)
        composite = CompositeJastrow([bh, cp])
        
        # Initialize composite params and copy from CPBH
        params_composite = composite.init_params()
        
        # Copy BH parameters from CPBH
        params_composite[0]['b_raw'] = params_cpbh['b_raw']
        params_composite[0]['d_raw'] = params_cpbh['d_raw']
        params_composite[0]['c_raw'] = params_cpbh['c_raw']
        
        # Copy CP parameters from CPBH
        params_composite[1]['cp_U'] = params_cpbh['cp_U']
        params_composite[1]['cp_C'] = params_cpbh['cp_C']
        params_composite[1]['cp_bias'] = params_cpbh['cp_bias']
        params_composite[1]['cp_scale'] = params_cpbh['cp_scale']
        
        # Compute values
        u_cpbh = cpbh._compute(self.r1, self.r2, params_cpbh)
        u_composite = composite._compute(self.r1, self.r2, params_composite)
        
        # Should match exactly
        np.testing.assert_allclose(u_cpbh, u_composite, rtol=1e-10, atol=1e-12,
                                   err_msg="CPBH and CompositeJastrow([BH, CP]) values don't match")
        
        print(f"✓ Values match: u_cpbh = {u_cpbh:.10f}, u_composite = {u_composite:.10f}")
    
    def test_same_gradient_with_same_params(self):
        """Test that ∇u is the same for CPBH and CompositeJastrow([BH, CP])."""
        # Create CPBH
        cpbh = CPBH(self.mol, cp_degree=self.cp_degree, cp_rank=self.cp_rank,
                    cp_scale_init=self.cp_scale_init)
        params_cpbh = cpbh.init_params(key=self.key)
        
        # BH supplies the cusp; CP supplies only the polynomial residual.
        bh = BoysHandy(self.mol)
        cp = CP(self.mol, cp_degree=self.cp_degree, cp_rank=self.cp_rank,
                cp_scale_init=self.cp_scale_init, use_raw_distances=False,
                include_cusp=False)
        composite = CompositeJastrow([bh, cp])
        
        # Initialize composite params and copy from CPBH
        params_composite = composite.init_params()
        
        # Copy parameters
        params_composite[0]['b_raw'] = params_cpbh['b_raw']
        params_composite[0]['d_raw'] = params_cpbh['d_raw']
        params_composite[0]['c_raw'] = params_cpbh['c_raw']
        params_composite[1]['cp_U'] = params_cpbh['cp_U']
        params_composite[1]['cp_C'] = params_cpbh['cp_C']
        params_composite[1]['cp_bias'] = params_cpbh['cp_bias']
        params_composite[1]['cp_scale'] = params_cpbh['cp_scale']
        
        # Compute gradients
        grad_cpbh = cpbh.grad_r(self.r1, self.r2, params_cpbh)
        grad_composite = composite.grad_r(self.r1, self.r2, params_composite)
        
        # Should match
        np.testing.assert_allclose(grad_cpbh, grad_composite, rtol=1e-8, atol=1e-10,
                                   err_msg="CPBH and CompositeJastrow([BH, CP]) gradients don't match")
        
        print(f"✓ Gradients match: max diff = {np.max(np.abs(grad_cpbh - grad_composite)):.2e}")
    
    def test_same_laplacian_with_same_params(self):
        """Test that ∇²u is the same for CPBH and CompositeJastrow([BH, CP])."""
        # Create CPBH
        cpbh = CPBH(self.mol, cp_degree=self.cp_degree, cp_rank=self.cp_rank,
                    cp_scale_init=self.cp_scale_init)
        params_cpbh = cpbh.init_params(key=self.key)
        
        # BH supplies the cusp; CP supplies only the polynomial residual.
        bh = BoysHandy(self.mol)
        cp = CP(self.mol, cp_degree=self.cp_degree, cp_rank=self.cp_rank,
                cp_scale_init=self.cp_scale_init, use_raw_distances=False,
                include_cusp=False)
        composite = CompositeJastrow([bh, cp])
        
        # Initialize composite params and copy from CPBH
        params_composite = composite.init_params()
        
        # Copy parameters
        params_composite[0]['b_raw'] = params_cpbh['b_raw']
        params_composite[0]['d_raw'] = params_cpbh['d_raw']
        params_composite[0]['c_raw'] = params_cpbh['c_raw']
        params_composite[1]['cp_U'] = params_cpbh['cp_U']
        params_composite[1]['cp_C'] = params_cpbh['cp_C']
        params_composite[1]['cp_bias'] = params_cpbh['cp_bias']
        params_composite[1]['cp_scale'] = params_cpbh['cp_scale']
        
        # Compute Laplacians
        lapl_cpbh = cpbh.laplacian_r(self.r1, self.r2, params_cpbh)
        lapl_composite = composite.laplacian_r(self.r1, self.r2, params_composite)
        
        # Should match
        np.testing.assert_allclose(lapl_cpbh, lapl_composite, rtol=1e-6, atol=1e-8,
                                   err_msg="CPBH and CompositeJastrow([BH, CP]) Laplacians don't match")
        
        print(f"✓ Laplacians match: lapl_cpbh = {lapl_cpbh:.6f}, lapl_composite = {lapl_composite:.6f}")


if __name__ == '__main__':
    unittest.main()

