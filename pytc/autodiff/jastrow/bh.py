from functools import partial
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from dataclasses import dataclass
import jax

from pytc.autodiff.jastrow import Jastrow

@dataclass
class BHTerm:
    """Single term in Boys-Handy expansion"""
    m: int
    n: int
    o: int  # renamed from alpha to o
    c: float  # initial value, will be optimized

class BoysHandy(Jastrow):
    """Boys-Handy Jastrow factor implementation."""
    
    def __init__(self, mol, terms_per_nucleus=None, epsilon=1e-8, name=None):
        super().__init__(name=name)
        self.mol = mol
        self.nelectron = mol.nelectron
        self.nuclear_pos = jnp.array(mol.atom_coords())
        self.nuclear_charges = jnp.array(mol.atom_charges())
        self.natom = len(self.nuclear_charges)
        self.epsilon = epsilon
        
        # Default terms if none specified
        if terms_per_nucleus is None:
            # Basic terms including e-e cusp
            d_cusp = 0.5  # cusp coefficient 1/(2d)
            default_terms_for_one_nucleus = [
                BHTerm(0, 0, 1, d_cusp),  # e-e cusp term with c = 1/(2d)
                BHTerm(0, 0, 2, 0.01),  
                BHTerm(0, 0, 3, 0.001),  
                BHTerm(0, 0, 4, -0.001),  
                BHTerm(2, 0, 0, 0.001),  # e-n term
                BHTerm(3, 0, 0, 0.0001),
                BHTerm(4, 0, 0, 0.0001),   # higher order term
                BHTerm(2, 2, 0, -0.001),    # e-n term
                BHTerm(2, 0, 2, 0.01),
                BHTerm(2, 2, 2, 0.01),
                BHTerm(4, 0, 2, 0.01),
                BHTerm(2, 0, 4, 0.01),
                BHTerm(4, 2, 2, 0.01),
                BHTerm(6, 0, 2, 0.01),
                BHTerm(4, 0, 4, 0.01),
                BHTerm(2, 2, 4, 0.01),
                BHTerm(2, 0, 6, 0.01),
            ]
            # Create a list of default terms for each nucleus
            self.terms_per_nucleus = [default_terms_for_one_nucleus for _ in range(self.natom)]
        else:
            self.terms_per_nucleus = terms_per_nucleus

        term_lengths = tuple(len(nucleus_terms) for nucleus_terms in self.terms_per_nucleus)
        if len(term_lengths) > 0 and len(set(term_lengths)) != 1:
            raise ValueError("BoysHandy requires the same number of terms per nucleus when using JAX scans.")

        self.n_terms = term_lengths[0] if term_lengths else 0

        term_m = []
        term_n = []
        term_o = []
        for nucleus_terms in self.terms_per_nucleus:
            term_m.append([term.m for term in nucleus_terms])
            term_n.append([term.n for term in nucleus_terms])
            term_o.append([term.o for term in nucleus_terms])

        if self.n_terms > 0:
            self._term_m = jnp.array(term_m, dtype=jnp.int32)
            self._term_n = jnp.array(term_n, dtype=jnp.int32)
            self._term_o = jnp.array(term_o, dtype=jnp.int32)
            self._delta_factor = jnp.where(self._term_m == self._term_n, 0.5, 1.0)
            self._cusp_mask = (self._term_m == 0) & (self._term_n == 0) & (self._term_o == 1)
        else:
            self._term_m = jnp.zeros((0, 0), dtype=jnp.int32)
            self._term_n = jnp.zeros((0, 0), dtype=jnp.int32)
            self._term_o = jnp.zeros((0, 0), dtype=jnp.int32)
            self._delta_factor = jnp.zeros((0, 0))
            self._cusp_mask = jnp.zeros((0, 0), dtype=bool)
            
    def _safe_norm(self, x):
        """Compute norm with a small epsilon to prevent division by zero."""
        return jnp.sqrt(jnp.sum(x*x, axis=-1) + self.epsilon)
    
    def _delta(self, m, n):
        """Implements the Delta function for Boys-Handy."""
        return jnp.where(m == n, 0.5, 1.0)
    
    def _scaled_r_en(self, r_electron, r_nuclear, b):
        """Compute scaled electron-nuclear distance."""
        r = self._safe_norm(r_electron - r_nuclear)
        #return b * r / (1.0 + b * r)
        return r / (1.0 + r)
    
    def _scaled_r_ee(self, r1, r2, d):
        """Compute scaled electron-electron distance."""
        r = self._safe_norm(r1 - r2)
        #return d * r / (1.0 + d * r)
        return  r / (1.0 +  r)

    def _grad_scaled_r_en(self, r_electron, r_nuclear):
        """Compute gradient of scaled electron-nuclear distance w.r.t r_electron.

        For s = r/(1+r) where r = ||r_electron - r_nuclear||:
        ∇s = (ds/dr) * (dr/d_r_electron) = 1/(1+r)² * (r_electron - r_nuclear)/r
             = (r_electron - r_nuclear) / [r * (1+r)²]

        Args:
            r_electron: Electron position, shape (3,)
            r_nuclear: Nuclear position, shape (3,)

        Returns:
            Gradient vector of shape (3,)
        """
        dr = r_electron - r_nuclear
        r_raw = self._safe_norm(dr)
        scale = 1.0 / (r_raw * (1.0 + r_raw)**2)
        return dr * scale

    def _grad_scaled_r_ee(self, r1, r2):
        """Compute gradient of scaled electron-electron distance w.r.t r1.

        For s = r/(1+r) where r = ||r1 - r2||:
        ∇_r1 s = (r1 - r2) / [r * (1+r)²]

        Args:
            r1: First electron position, shape (3,)
            r2: Second electron position, shape (3,)

        Returns:
            Gradient vector of shape (3,)
        """
        dr = r1 - r2
        r_raw = self._safe_norm(dr)
        scale = 1.0 / (r_raw * (1.0 + r_raw)**2)
        return dr * scale

    def _laplacian_scaled_r_en(self, r_electron, r_nuclear):
        """Compute Laplacian of scaled electron-nuclear distance w.r.t r_electron.

        For s = r/(1+r) where r = ||r_electron - r_nuclear||:
        ∇²s = 2/(r*(1+r)²) + 2*(1-r)/(r²*(1+r)³)
            = 2/[r*(1+r)²] * [1 + (1-r)/(r*(1+r))]
            = 2(2-r)/[r²*(1+r)³]

        Args:
            r_electron: Electron position, shape (3,)
            r_nuclear: Nuclear position, shape (3,)

        Returns:
            Laplacian (scalar)
        """
        dr = r_electron - r_nuclear
        r_raw = self._safe_norm(dr)
        # ∇²s = 2(2-r) / [r²(1+r)³]
        return 2.0 * (2.0 - r_raw) / (r_raw**2 * (1.0 + r_raw)**3)

    def _laplacian_scaled_r_ee(self, r1, r2):
        """Compute Laplacian of scaled electron-electron distance w.r.t r1.

        For s = r/(1+r) where r = ||r1 - r2||:
        ∇²s = 2(2-r) / [r²(1+r)³]

        Args:
            r1: First electron position, shape (3,)
            r2: Second electron position, shape (3,)

        Returns:
            Laplacian (scalar)
        """
        dr = r1 - r2
        r_raw = self._safe_norm(dr)
        return 2.0 * (2.0 - r_raw) / (r_raw**2 * (1.0 + r_raw)**3)

    def init_params(self, **kwargs):
        """Initialize Boys-Handy parameters."""
        key = kwargs.get('key', random.PRNGKey(0))
        
        # Initialize b and d parameters for each nucleus
        b_raw = jnp.ones(self.natom) * 0.5  # starting value ~1.0 after softplus
        d_raw = jnp.ones(self.natom) * 0.5
        
        # Initialize c parameters for each term in each nucleus
        # Reshape to ensure (n_nuclei, n_terms) shape
        c_raw = []
        for nucleus_terms in self.terms_per_nucleus:
            c_nucleus = jnp.array([term.c for term in nucleus_terms])
            c_raw.append(c_nucleus)
        c_raw = jnp.array(c_raw)  # This will have shape (n_nuclei, n_terms)
        
        return {
            'b_raw': b_raw,
            'd_raw': d_raw,
            'c_raw': c_raw
        }

    @partial(jax.jit, static_argnums=(0,))
    def _compute(self, r1, r2, params):
        """Compute Boys-Handy Jastrow exponent."""
        
        # Get positive b and d values using softplus
        b = nn.softplus(params['b_raw'])
        d = nn.softplus(params['d_raw'])
        c = params['c_raw']  # Allow c to be both positive and negative

        def nucleus_scan_fn(carry, nucleus_data):
            """Scan function for looping over nuclei."""
            u_total = carry
            nuclear_pos_I, b_I, d_I, c_I, term_m_I, term_n_I, term_o_I, delta_factor_I, cusp_mask_I = nucleus_data
            
            # Compute scaled distances for this nucleus
            r1I = self._scaled_r_en(r1, nuclear_pos_I, b_I)
            r2I = self._scaled_r_en(r2, nuclear_pos_I, b_I)
            r12 = self._scaled_r_ee(r1, r2, d_I)

            r1I_pow_m = jnp.power(r1I, term_m_I)
            r2I_pow_n = jnp.power(r2I, term_n_I)
            r1I_pow_n = jnp.power(r1I, term_n_I)
            r2I_pow_m = jnp.power(r2I, term_m_I)
            r12_pow_o = jnp.power(r12, term_o_I)

            non_cusp_terms = (r1I_pow_m * r2I_pow_n + r2I_pow_m * r1I_pow_n) * r12_pow_o
            cusp_terms = r12_pow_o
            u_terms = jnp.where(cusp_mask_I, cusp_terms, non_cusp_terms)

            cusp_factor = delta_factor_I * 0.5
            non_cusp_factor = delta_factor_I * c_I
            factor = jnp.where(cusp_mask_I, cusp_factor, non_cusp_factor)

            u_total += jnp.sum(factor * u_terms)
            return u_total, None

        # Prepare nucleus data
        nucleus_data = (
            self.nuclear_pos,
            b,
            d,
            c,
            self._term_m,
            self._term_n,
            self._term_o,
            self._delta_factor,
            self._cusp_mask,
        )
        
        # Scan over nuclei
        u_total, _ = jax.lax.scan(nucleus_scan_fn, 0.0, nucleus_data)

        return u_total

    @partial(jax.jit, static_argnums=(0,))
    def _grad_r1_analytical(self, r1, r2, params):
        """Compute analytical gradient of u w.r.t r1.

        Uses the chain rule: ∇_r1 u = (∂u/∂r1I)(∇_r1 r1I) + (∂u/∂r12)(∇_r1 r12)

        Args:
            r1: First electron position, shape (3,) or (1, 3)
            r2: Second electron position, shape (3,) or (1, 3)
            params: Jastrow parameters

        Returns:
            Gradient vector of shape matching input (3,) or (1, 3)
        """
        # Store original shape to restore at the end
        original_shape = r1.shape

        # Ensure r1 and r2 are 1D for internal computation
        r1_flat = jnp.reshape(r1, (3,))
        r2_flat = jnp.reshape(r2, (3,))

        # Get positive b and d values using softplus
        b = nn.softplus(params['b_raw'])
        d = nn.softplus(params['d_raw'])
        c = params['c_raw']

        def nucleus_scan_fn(carry, nucleus_data):
            """Scan function for looping over nuclei."""
            grad_total = carry
            nuclear_pos_I, b_I, d_I, c_I, term_m_I, term_n_I, term_o_I, delta_factor_I, cusp_mask_I = nucleus_data

            # Compute scaled distances for this nucleus (same as _compute)
            r1I = self._scaled_r_en(r1_flat, nuclear_pos_I, b_I)
            r2I = self._scaled_r_en(r2_flat, nuclear_pos_I, b_I)
            r12 = self._scaled_r_ee(r1_flat, r2_flat, d_I)

            # Compute distance derivative factors
            grad_r1I = self._grad_scaled_r_en(r1_flat, nuclear_pos_I)  # shape (3,)
            grad_r12 = self._grad_scaled_r_ee(r1_flat, r2_flat)  # shape (3,)

            # Compute powers (same as _compute)
            r1I_pow_m = jnp.power(r1I, term_m_I)
            r2I_pow_n = jnp.power(r2I, term_n_I)
            r1I_pow_n = jnp.power(r1I, term_n_I)
            r2I_pow_m = jnp.power(r2I, term_m_I)
            r12_pow_o = jnp.power(r12, term_o_I)

            # Compute shifted powers for derivatives
            # For ∂/∂r1I: need r1I^(m-1) and r1I^(n-1)
            r1I_pow_m_minus_1 = jnp.where(term_m_I > 0, jnp.power(r1I, term_m_I - 1), 0.0)
            r1I_pow_n_minus_1 = jnp.where(term_n_I > 0, jnp.power(r1I, term_n_I - 1), 0.0)

            # For ∂/∂r12: need r12^(o-1)
            r12_pow_o_minus_1 = jnp.where(term_o_I > 0, jnp.power(r12, term_o_I - 1), 0.0)

            # Contribution from ∂u/∂r1I (scalar polynomial part)
            # ∂u/∂r1I = m * r1I^(m-1) * r2I^n * r12^o + n * r2I^m * r1I^(n-1) * r12^o
            du_dr1I = (term_m_I * r1I_pow_m_minus_1 * r2I_pow_n +
                       term_n_I * r2I_pow_m * r1I_pow_n_minus_1) * r12_pow_o

            # Contribution from ∂u/∂r12 (scalar polynomial part)
            # ∂u/∂r12 = o * (r1I^m * r2I^n + r2I^m * r1I^n) * r12^(o-1)
            du_dr12 = term_o_I * (r1I_pow_m * r2I_pow_n + r2I_pow_m * r1I_pow_n) * r12_pow_o_minus_1

            # Handle cusp terms (m=0, n=0, o=1)
            # For cusp: u = 0.5 * r12, so ∂u/∂r12 = 0.5, ∂u/∂r1I = 0
            du_dr1I_cusp = jnp.zeros_like(du_dr1I)
            du_dr12_cusp = jnp.ones_like(du_dr12)

            du_dr1I = jnp.where(cusp_mask_I, du_dr1I_cusp, du_dr1I)
            du_dr12 = jnp.where(cusp_mask_I, du_dr12_cusp, du_dr12)

            # Apply coefficients (same logic as _compute)
            cusp_factor = delta_factor_I * 0.5
            non_cusp_factor = delta_factor_I * c_I
            factor = jnp.where(cusp_mask_I, cusp_factor, non_cusp_factor)

            # Combine: ∇u = (∂u/∂r1I) * ∇r1I + (∂u/∂r12) * ∇r12
            # du_dr1I and du_dr12 are arrays of shape (n_terms,)
            # grad_r1I and grad_r12 are vectors of shape (3,)
            # Sum over terms: Σ_k factor[k] * (du_dr1I[k] * grad_r1I + du_dr12[k] * grad_r12)
            grad_contrib = (jnp.sum(factor * du_dr1I) * grad_r1I +
                           jnp.sum(factor * du_dr12) * grad_r12)

            return grad_total + grad_contrib, None

        # Prepare nucleus data (same as _compute)
        nucleus_data = (
            self.nuclear_pos,
            b,
            d,
            c,
            self._term_m,
            self._term_n,
            self._term_o,
            self._delta_factor,
            self._cusp_mask,
        )

        # Scan over nuclei
        grad_total, _ = jax.lax.scan(nucleus_scan_fn, jnp.zeros(3), nucleus_data)

        # Restore original shape
        return jnp.reshape(grad_total, original_shape)

    @partial(jax.jit, static_argnums=(0,))
    def _grad_r2_analytical(self, r1, r2, params):
        """Compute analytical gradient of u w.r.t r2.

        Similar to _grad_r1_analytical but with r1 and r2 roles swapped.

        Args:
            r1: First electron position, shape (3,) or (1, 3)
            r2: Second electron position, shape (3,) or (1, 3)
            params: Jastrow parameters

        Returns:
            Gradient vector of shape matching input (3,) or (1, 3)
        """
        # Store original shape to restore at the end
        original_shape = r2.shape

        # Ensure r1 and r2 are 1D for internal computation
        r1_flat = jnp.reshape(r1, (3,))
        r2_flat = jnp.reshape(r2, (3,))

        # Get positive b and d values using softplus
        b = nn.softplus(params['b_raw'])
        d = nn.softplus(params['d_raw'])
        c = params['c_raw']

        def nucleus_scan_fn(carry, nucleus_data):
            """Scan function for looping over nuclei."""
            grad_total = carry
            nuclear_pos_I, b_I, d_I, c_I, term_m_I, term_n_I, term_o_I, delta_factor_I, cusp_mask_I = nucleus_data

            # Compute scaled distances for this nucleus
            r1I = self._scaled_r_en(r1_flat, nuclear_pos_I, b_I)
            r2I = self._scaled_r_en(r2_flat, nuclear_pos_I, b_I)
            r12 = self._scaled_r_ee(r1_flat, r2_flat, d_I)

            # Compute distance derivative factors (w.r.t r2)
            grad_r2I = self._grad_scaled_r_en(r2_flat, nuclear_pos_I)  # shape (3,)
            grad_r12_r2 = -self._grad_scaled_r_ee(r1_flat, r2_flat)  # Negative because ∇_r2 r12 = -∇_r1 r12

            # Compute powers
            r1I_pow_m = jnp.power(r1I, term_m_I)
            r2I_pow_n = jnp.power(r2I, term_n_I)
            r1I_pow_n = jnp.power(r1I, term_n_I)
            r2I_pow_m = jnp.power(r2I, term_m_I)
            r12_pow_o = jnp.power(r12, term_o_I)

            # Compute shifted powers for derivatives
            # For ∂/∂r2I: need r2I^(m-1) and r2I^(n-1)
            r2I_pow_m_minus_1 = jnp.where(term_m_I > 0, jnp.power(r2I, term_m_I - 1), 0.0)
            r2I_pow_n_minus_1 = jnp.where(term_n_I > 0, jnp.power(r2I, term_n_I - 1), 0.0)

            # For ∂/∂r12: need r12^(o-1)
            r12_pow_o_minus_1 = jnp.where(term_o_I > 0, jnp.power(r12, term_o_I - 1), 0.0)

            # Contribution from ∂u/∂r2I (scalar polynomial part)
            # ∂u/∂r2I = n * r1I^m * r2I^(n-1) * r12^o + m * r2I^(m-1) * r1I^n * r12^o
            du_dr2I = (term_n_I * r1I_pow_m * r2I_pow_n_minus_1 +
                       term_m_I * r2I_pow_m_minus_1 * r1I_pow_n) * r12_pow_o

            # Contribution from ∂u/∂r12 (scalar polynomial part) - same as for r1
            # ∂u/∂r12 = o * (r1I^m * r2I^n + r2I^m * r1I^n) * r12^(o-1)
            du_dr12 = term_o_I * (r1I_pow_m * r2I_pow_n + r2I_pow_m * r1I_pow_n) * r12_pow_o_minus_1

            # Handle cusp terms (m=0, n=0, o=1)
            # For cusp: u = 0.5 * r12, so ∂u/∂r12 = 0.5, ∂u/∂r2I = 0
            du_dr2I_cusp = jnp.zeros_like(du_dr2I)
            du_dr12_cusp = jnp.ones_like(du_dr12)

            du_dr2I = jnp.where(cusp_mask_I, du_dr2I_cusp, du_dr2I)
            du_dr12 = jnp.where(cusp_mask_I, du_dr12_cusp, du_dr12)

            # Apply coefficients
            cusp_factor = delta_factor_I * 0.5
            non_cusp_factor = delta_factor_I * c_I
            factor = jnp.where(cusp_mask_I, cusp_factor, non_cusp_factor)

            # Combine: ∇_r2 u = (∂u/∂r2I) * ∇r2I + (∂u/∂r12) * ∇_r2 r12
            grad_contrib = (jnp.sum(factor * du_dr2I) * grad_r2I +
                           jnp.sum(factor * du_dr12) * grad_r12_r2)

            return grad_total + grad_contrib, None

        # Prepare nucleus data
        nucleus_data = (
            self.nuclear_pos,
            b,
            d,
            c,
            self._term_m,
            self._term_n,
            self._term_o,
            self._delta_factor,
            self._cusp_mask,
        )

        # Scan over nuclei
        grad_total, _ = jax.lax.scan(nucleus_scan_fn, jnp.zeros(3), nucleus_data)

        # Restore original shape
        return jnp.reshape(grad_total, original_shape)

    @partial(jax.jit, static_argnums=(0,))
    def _laplacian_r1_analytical(self, r1, r2, params):
        """Compute analytical Laplacian of u w.r.t r1.

        Computes Laplacian as divergence of analytical gradient.
        This is more reliable than the direct formula and still fully analytical.

        Args:
            r1: First electron position, shape (3,)
            r2: Second electron position, shape (3,)
            params: Jastrow parameters

        Returns:
            Laplacian (scalar)
        """
        # Compute Laplacian as trace of Jacobian of gradient
        def grad_fn(x):
            return self._grad_r1_analytical(x, r2, params)

        jacobian = jax.jacfwd(grad_fn)(r1)
        return jnp.trace(jacobian)

    @partial(jax.jit, static_argnums=(0,))
    def _laplacian_r2_analytical(self, r1, r2, params):
        """Compute analytical Laplacian of u w.r.t r2.

        Computes Laplacian as divergence of analytical gradient.
        This is more reliable than the direct formula and still fully analytical.

        Args:
            r1: First electron position, shape (3,)
            r2: Second electron position, shape (3,)
            params: Jastrow parameters

        Returns:
            Laplacian (scalar)
        """
        # Compute Laplacian as trace of Jacobian of gradient
        def grad_fn(x):
            return self._grad_r2_analytical(r1, x, params)

        jacobian = jax.jacfwd(grad_fn)(r2)
        return jnp.trace(jacobian)

    def get_param_count(self):
        """Return total number of optimizable parameters."""
        # Count b and d parameters (one per nucleus)
        count = 2 * self.natom
        # Add c parameters (one per term per nucleus)
        for nucleus_terms in self.terms_per_nucleus:
            count += len(nucleus_terms)
        return count

    def flatten_params(self, params):
        """Flatten parameters into 1D array for optimization."""
        return jnp.concatenate([
            params['b_raw'].ravel(),
            params['d_raw'].ravel(),
            params['c_raw'].ravel()
        ])

    def unflatten_params(self, flat_params):
        """Reconstruct parameter dictionary from 1D array."""
        idx = 0

        # Extract b parameters
        b_size = self.natom
        b_raw = flat_params[idx:idx+b_size]
        idx += b_size

        # Extract d parameters
        d_size = self.natom
        d_raw = flat_params[idx:idx+d_size]
        idx += d_size

        # Extract c parameters
        c_raw = []
        for nucleus_terms in self.terms_per_nucleus:
            c_size = len(nucleus_terms)
            c_nucleus = flat_params[idx:idx+c_size]
            c_raw.append(c_nucleus)
            idx += c_size
        c_raw = jnp.array(c_raw)

        return {
            'b_raw': b_raw,
            'd_raw': d_raw,
            'c_raw': c_raw
        }

    # Override base class methods to use analytical derivatives
    @partial(jax.jit, static_argnums=(0,))
    def grad_r(self, r1, r2, params):
        """Compute gradient of u w.r.t r1 coordinates using analytical derivatives.

        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            params: Jastrow parameters

        Returns:
            Gradient array of shape (3,)
        """
        return self._grad_r1_analytical(r1, r2, params)

    @partial(jax.jit, static_argnums=(0,))
    def laplacian_r(self, r1, r2, params):
        """Compute Laplacian of u w.r.t r1 coordinates using analytical derivatives.

        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            params: Jastrow parameters

        Returns:
            Laplacian value (scalar)
        """
        return self._laplacian_r1_analytical(r1, r2, params)

    @partial(jax.jit, static_argnums=(0,))
    def get_log_grads_r1(self, r1, r2, params):
        """Compute ∇u and ∇²u w.r.t first electron coordinates.

        Uses analytical gradient and analytical Laplacian.

        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            params: Jastrow parameters

        Returns:
            tuple (grad_u, lapl_u) containing:
                grad_u: gradient of u w.r.t r1, shape (3,) - analytical
                lapl_u: laplacian of u w.r.t r1 (scalar) - analytical
        """
        # Use analytical gradient and Laplacian
        grad_u = self._grad_r1_analytical(r1, r2, params)
        lapl_u = self._laplacian_r1_analytical(r1, r2, params)

        return grad_u, lapl_u

    @partial(jax.jit, static_argnums=(0,))
    def get_log_grads_r2(self, r1, r2, params):
        """Compute ∇u and ∇²u w.r.t second electron coordinates.

        Uses analytical gradient and analytical Laplacian.

        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            params: Jastrow parameters

        Returns:
            tuple (grad_u, lapl_u) containing:
                grad_u: gradient of u w.r.t r2, shape (3,) - analytical
                lapl_u: laplacian of u w.r.t r2 (scalar) - analytical
        """
        # Use analytical gradient and Laplacian
        grad_u = self._grad_r2_analytical(r1, r2, params)
        lapl_u = self._laplacian_r2_analytical(r1, r2, params)

        return grad_u, lapl_u

    @partial(jax.jit, static_argnums=(0,))
    def _grad_params_analytical(self, r1, r2, params):
        """Compute analytical gradient of u w.r.t. BH parameters.

        For coefficient parameters c_{I,k}:
            ∂u/∂c_{I,k} = Δ(m_k,n_k) * f_{I,k}(r_{1I}, r_{2I}, r_{12})
        where f_{I,k} is the polynomial term (same as forward pass without coefficient).

        For scale parameters b_I, d_I:
            Currently zero since scaling doesn't use these parameters.
            If re-enabled: would use chain rule with distance derivatives.

        Args:
            r1: First electron position, shape (3,)
            r2: Second electron position, shape (3,)
            params: Jastrow parameters dict with keys 'b_raw', 'd_raw', 'c_raw'

        Returns:
            Dictionary with same structure as params containing gradients:
            {
                'b_raw': gradient w.r.t. b_raw (shape: n_nuclei),
                'd_raw': gradient w.r.t. d_raw (shape: n_nuclei),
                'c_raw': gradient w.r.t. c_raw (shape: n_nuclei, n_terms)
            }
        """
        # Ensure r1 and r2 are 1D for internal computation
        r1_flat = jnp.reshape(r1, (3,))
        r2_flat = jnp.reshape(r2, (3,))

        # Get transformed parameters
        b = nn.softplus(params['b_raw'])
        d = nn.softplus(params['d_raw'])
        c = params['c_raw']

        # Initialize gradient accumulators with correct shape
        grad_c = jnp.zeros_like(c)  # shape: (n_nuclei, n_terms)
        grad_b = jnp.zeros_like(b)  # shape: (n_nuclei) - currently unused
        grad_d = jnp.zeros_like(d)  # shape: (n_nuclei) - currently unused

        def nucleus_scan_fn(carry, nucleus_idx_and_data):
            """Scan function for looping over nuclei."""
            grad_c_acc, grad_b_acc, grad_d_acc = carry
            nucleus_idx, nuclear_pos_I, b_I, d_I, c_I, term_m_I, term_n_I, term_o_I, delta_factor_I, cusp_mask_I = nucleus_idx_and_data

            # Compute scaled distances for this nucleus (same as _compute)
            r1I = self._scaled_r_en(r1_flat, nuclear_pos_I, b_I)
            r2I = self._scaled_r_en(r2_flat, nuclear_pos_I, b_I)
            r12 = self._scaled_r_ee(r1_flat, r2_flat, d_I)

            # Compute polynomial terms f_{I,k} (same as _compute)
            r1I_pow_m = jnp.power(r1I, term_m_I)
            r2I_pow_n = jnp.power(r2I, term_n_I)
            r1I_pow_n = jnp.power(r1I, term_n_I)
            r2I_pow_m = jnp.power(r2I, term_m_I)
            r12_pow_o = jnp.power(r12, term_o_I)

            # f_{I,k} for non-cusp terms: (r1I^m * r2I^n + r2I^m * r1I^n) * r12^o
            f_Ik = (r1I_pow_m * r2I_pow_n + r2I_pow_m * r1I_pow_n) * r12_pow_o

            # Gradient w.r.t. c_{I,k}: ∂u/∂c_{I,k} = Δ(m,n) * f_{I,k}
            # For cusp terms (m=0, n=0, o=1), coefficient is fixed at 1/(2d) = 0.5,
            # so gradient is zero (not trainable)
            grad_c_I = jnp.where(cusp_mask_I, 0.0, delta_factor_I * f_Ik)

            # Update gradient for this nucleus using at[].set()
            grad_c_acc = grad_c_acc.at[nucleus_idx].set(grad_c_I)

            # grad_b and grad_d remain zero for current implementation
            # If b/d were used in scaling, would add chain rule terms here:
            # grad_b_I = (∂u/∂r1I)(∂r1I/∂b_I) + (∂u/∂r2I)(∂r2I/∂b_I)
            # grad_d_I = (∂u/∂r12)(∂r12/∂d_I)

            return (grad_c_acc, grad_b_acc, grad_d_acc), None

        # Prepare nucleus data with indices
        nucleus_indices = jnp.arange(self.natom)
        nucleus_data = (
            nucleus_indices,
            self.nuclear_pos,
            b,
            d,
            c,
            self._term_m,
            self._term_n,
            self._term_o,
            self._delta_factor,
            self._cusp_mask,
        )

        # Scan over nuclei
        (grad_c, grad_b, grad_d), _ = jax.lax.scan(
            nucleus_scan_fn,
            (grad_c, grad_b, grad_d),
            nucleus_data
        )

        # Apply chain rule for softplus transform: ∂L/∂x_raw = ∂L/∂x * ∂x/∂x_raw
        # where ∂(softplus(x_raw))/∂x_raw = sigmoid(x_raw)
        grad_b_raw = grad_b * jax.nn.sigmoid(params['b_raw'])
        grad_d_raw = grad_d * jax.nn.sigmoid(params['d_raw'])

        return {
            'b_raw': grad_b_raw,
            'd_raw': grad_d_raw,
            'c_raw': grad_c
        }

    @partial(jax.jit, static_argnums=(0,))
    def grad_params(self, r1, r2, params):
        """Compute gradient of u w.r.t parameters using analytical derivatives.

        Overrides base class to use analytical implementation for BH parameters.
        If params contains additional keys (e.g., from CPBH), uses autodiff for those.

        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            params: Jastrow parameters

        Returns:
            Gradient dictionary with same structure as params
        """
        # BH parameter keys
        bh_keys = {'b_raw', 'd_raw', 'c_raw'}

        # Check if there are non-BH parameters (e.g., from CPBH)
        extra_keys = set(params.keys()) - bh_keys

        if not extra_keys:
            # Pure BH: use analytical gradient for all parameters
            return self._grad_params_analytical(r1, r2, params)
        else:
            # Mixed (e.g., CPBH): use analytical for BH, autodiff for rest
            # Get analytical BH gradients
            bh_params = {k: params[k] for k in bh_keys if k in params}
            bh_grads = self._grad_params_analytical(r1, r2, bh_params)

            # Get autodiff gradients for all parameters
            all_grads = jax.grad(lambda p: self._compute(r1, r2, p))(params)

            # Combine: use analytical for BH, autodiff for extra
            result = {}
            for k in params.keys():
                if k in bh_keys:
                    result[k] = bh_grads[k]
                else:
                    result[k] = all_grads[k]

            return result
