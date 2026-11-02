"""Test analytical parameter gradients for Boys-Handy Jastrow factor."""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from pyscf import gto

from pytc.autodiff.jastrow.bh import BoysHandy


def test_bh_param_grad_vs_autodiff():
    """Test that analytical parameter gradient matches autodiff."""
    # Create a simple molecule
    mol = gto.M(
        atom='Be 0 0 0',
        basis='sto-3g',
        unit='Bohr'
    )
    
    # Create BH Jastrow
    jastrow = BoysHandy(mol)
    params = jastrow.init_params()
    
    # Random electron positions
    key = random.PRNGKey(42)
    key, subkey1, subkey2 = random.split(key, 3)
    r1 = random.normal(subkey1, (3,)) * 0.5
    r2 = random.normal(subkey2, (3,)) * 0.5
    
    # Compute analytical gradient
    grad_analytical = jastrow.grad_params(r1, r2, params)
    
    # Compute autodiff gradient (using base class method directly)
    grad_autodiff = jax.grad(lambda p: jastrow._compute(r1, r2, p))(params)
    
    # Compare gradients
    print("\n=== Analytical vs Autodiff Parameter Gradients ===")
    print(f"grad_c analytical shape: {grad_analytical['c_raw'].shape}")
    print(f"grad_c autodiff shape: {grad_autodiff['c_raw'].shape}")
    
    # Check c_raw gradients (the important ones)
    c_diff = jnp.abs(grad_analytical['c_raw'] - grad_autodiff['c_raw'])
    c_max_diff = jnp.max(c_diff)
    c_mean_diff = jnp.mean(c_diff)
    
    print(f"\nc_raw gradient differences:")
    print(f"  Max absolute difference: {c_max_diff:.2e}")
    print(f"  Mean absolute difference: {c_mean_diff:.2e}")
    print(f"  Analytical c_raw grad (first nucleus, first 5 terms):")
    print(f"    {grad_analytical['c_raw'][0, :5]}")
    print(f"  Autodiff c_raw grad (first nucleus, first 5 terms):")
    print(f"    {grad_autodiff['c_raw'][0, :5]}")
    
    # Check b_raw and d_raw gradients (should be zero for both)
    b_diff = jnp.abs(grad_analytical['b_raw'] - grad_autodiff['b_raw'])
    d_diff = jnp.abs(grad_analytical['d_raw'] - grad_autodiff['d_raw'])
    
    print(f"\nb_raw gradient differences:")
    print(f"  Max absolute difference: {jnp.max(b_diff):.2e}")
    print(f"  Analytical: {grad_analytical['b_raw']}")
    print(f"  Autodiff: {grad_autodiff['b_raw']}")
    
    print(f"\nd_raw gradient differences:")
    print(f"  Max absolute difference: {jnp.max(d_diff):.2e}")
    print(f"  Analytical: {grad_analytical['d_raw']}")
    print(f"  Autodiff: {grad_autodiff['d_raw']}")
    
    # Assert close match (machine precision)
    assert c_max_diff < 1e-6, f"c_raw gradient mismatch: max diff = {c_max_diff}"
    assert jnp.max(b_diff) < 1e-6, f"b_raw gradient mismatch"
    assert jnp.max(d_diff) < 1e-6, f"d_raw gradient mismatch"
    
    print("\n✓ All parameter gradients match within tolerance!")


def test_bh_param_grad_multiple_positions():
    """Test parameter gradients at multiple random positions."""
    mol = gto.M(
        atom='Be 0 0 0',
        basis='sto-3g',
        unit='Bohr'
    )
    
    jastrow = BoysHandy(mol)
    params = jastrow.init_params()
    
    key = random.PRNGKey(123)
    n_tests = 10
    max_diffs = []
    
    for i in range(n_tests):
        key, subkey1, subkey2 = random.split(key, 3)
        r1 = random.normal(subkey1, (3,)) * 0.5
        r2 = random.normal(subkey2, (3,)) * 0.5
        
        grad_analytical = jastrow.grad_params(r1, r2, params)
        grad_autodiff = jax.grad(lambda p: jastrow._compute(r1, r2, p))(params)
        
        c_diff = jnp.max(jnp.abs(grad_analytical['c_raw'] - grad_autodiff['c_raw']))
        max_diffs.append(c_diff)
    
    print(f"\n=== Multiple Position Tests ===")
    print(f"Tested {n_tests} random positions")
    print(f"Max c_raw gradient difference across all tests: {max(max_diffs):.2e}")
    print(f"Mean c_raw gradient difference: {np.mean(max_diffs):.2e}")
    
    assert max(max_diffs) < 1e-6, "Parameter gradients don't match across multiple positions"
    print("✓ All tests passed!")


if __name__ == "__main__":
    test_bh_param_grad_vs_autodiff()
    test_bh_param_grad_multiple_positions()

