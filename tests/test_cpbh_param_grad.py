"""Test that CPBH parameter gradients work correctly with analytical BH param grad."""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from pyscf import gto

from pytc.autodiff.jastrow.cpbh import CPBH


def test_cpbh_param_grad():
    """Test that CPBH parameter gradient works with analytical BH component."""
    print("\n" + "="*70)
    print("TEST: CPBH Parameter Gradients (BH analytical + CP autodiff)")
    print("="*70)
    
    # Create molecule
    mol = gto.M(
        atom='Be 0 0 0',
        basis='sto-3g',
        unit='Bohr'
    )
    
    # Create CPBH Jastrow
    jastrow = CPBH(mol, cp_rank=4, cp_degree=3, use_cp_residual=True)
    params = jastrow.init_params()

    # Random electron positions
    key = random.PRNGKey(42)
    key, subkey1, subkey2 = random.split(key, 3)
    r1 = random.normal(subkey1, (3,)) * 0.5
    r2 = random.normal(subkey2, (3,)) * 0.5

    print(f"\nMolecule: {mol.atom}")
    print(f"CP rank: {jastrow.cp_rank}")
    print(f"CP degree: {jastrow.cp_degree}")
    print(f"Use CP residual: {jastrow.use_cp_residual}")
    print(f"\nParameter keys: {list(params.keys())}")
    
    # Compute parameter gradient (should use analytical BH + autodiff CP)
    print("\nComputing CPBH parameter gradient...")
    grad = jastrow.grad_params(r1, r2, params)

    # Check structure
    print("\nGradient structure:")
    print(f"  Keys: {list(grad.keys())}")
    print(f"  BH c_raw shape: {grad['c_raw'].shape}")
    print(f"  BH b_raw shape: {grad['b_raw'].shape}")
    print(f"  BH d_raw shape: {grad['d_raw'].shape}")
    if 'cp_U' in grad:
        print(f"  CP U shape: {grad['cp_U'].shape}")
        print(f"  CP C shape: {grad['cp_C'].shape}")
        print(f"  CP bias shape: {grad['cp_bias'].shape}")

    # Verify BH gradient is non-zero (analytical should work)
    bh_grad_norm = jnp.linalg.norm(grad['c_raw'])
    print(f"\nBH c_raw gradient norm: {bh_grad_norm:.6f}")
    assert bh_grad_norm > 0, "BH gradient should be non-zero"

    # Verify CP gradient is non-zero (if CP is enabled)
    if 'cp_C' in grad:
        cp_grad_norm = jnp.linalg.norm(grad['cp_C'])
        print(f"CP C gradient norm: {cp_grad_norm:.6f}")
        assert cp_grad_norm > 0, "CP gradient should be non-zero"

    # Compare with full autodiff
    print("\nComparing with full autodiff...")
    grad_autodiff = jax.grad(lambda p: jastrow._compute(r1, r2, p))(params)

    # Check BH gradients match
    bh_diff = jnp.max(jnp.abs(grad['c_raw'] - grad_autodiff['c_raw']))
    print(f"BH c_raw max difference: {bh_diff:.2e}")
    assert bh_diff < 1e-6, f"BH gradient mismatch: {bh_diff}"

    # Check CP gradients match (if CP is enabled)
    if 'cp_C' in grad:
        cp_diff = jnp.max(jnp.abs(grad['cp_C'] - grad_autodiff['cp_C']))
        print(f"CP C max difference: {cp_diff:.2e}")
        assert cp_diff < 1e-6, f"CP gradient mismatch: {cp_diff}"
    
    print("\n✓ CPBH parameter gradients work correctly!")
    print("  - BH component uses analytical gradient")
    print("  - CP component uses autodiff")
    print("  - Both match full autodiff to machine precision")
    print("="*70)


if __name__ == "__main__":
    test_cpbh_param_grad()

