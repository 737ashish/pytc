"""
CP: Standalone Canonical Polyadic polynomial Jastrow factor.

This class implements a standalone CP polynomial network Jastrow factor with
per-nucleus networks. It can be combined with other Jastrow factors using
CompositeJastrow.

For each nucleus I:
- CP network: Polynomial on r² features (and optionally r³) for cusp-free learning

Architecture:
- Single network with [r1I², r2I², r12²] or [r1I², r2I², r12², r1I³, r2I³, r12³]
- Manual symmetrization: u = (CP(r1,r2) + CP(r2,r1)) / 2 (default)
- Or symmetric sums: [r1I²+r2I², r1I²*r2I², r12²] (if use_manual_symmetrization=False)

The CP recursion follows the pattern from qpolynets:
    out = U1(x)
    for i in 2..degree:
        out = Ui(x) * out + out
    result = C(out) + bias

Usage with CompositeJastrow:
    # CP + Boys-Handy (equivalent to CPBH)
    bh = BoysHandy(mol)
    cp = CP(mol, cp_degree=3, cp_rank=4)
    composite = CompositeJastrow([bh, cp])
    
    # CP + REXP (CPREXP)
    rexp = REXP()
    cp = CP(mol, cp_degree=3, cp_rank=4, use_raw_distances=True)
    composite = CompositeJastrow([rexp, cp])
"""
from functools import partial
from typing import Dict, Any, Optional

import jax
import jax.numpy as jnp
from jax import random

from .jastrow import Jastrow


class CP(Jastrow):
    """Standalone CP polynomial Jastrow factor with per-nucleus networks.

    Computes: u = cp_scale * (u_cusp + u_CP)  if include_cusp=True
              u = cp_scale * u_CP              if include_cusp=False

    Where:
    - u_cusp = 0.5 * r̄12 (electron-electron cusp term, coefficient 0.5 is fixed)
    - r̄12 = r12/(1+r12) (BH-style scaled distance)
    - u_CP = Σ_I u_cp^I(r1I, r2I, r12) (CP polynomial network)
    - I indexes nuclei
    - u_cp^I is a CP polynomial network for nucleus I
    - Features: [r1I², r2I², r12²] or [r1I², r2I², r12², r1I³, r2I³, r12³]
    - Manual symmetrization: u = (CP(r1,r2) + CP(r2,r1)) / 2

    Args:
        mol: PySCF molecule
        cp_degree: Polynomial degree for CP recursion (default: 3)
        cp_rank: CP rank (default: 4)
        cp_scale_init: Initial scale for CP contribution (default: 1.0)
        learnable_cp_scale: Make cp_scale learnable (default: True)
        use_log_scale: Use logarithmic parameterization for cp_scale (default: False)
        use_cubic_features: Include r³ features for complete degree coverage (default: False)
                           If True: [r1I², r2I², r12², r1I³, r2I³, r12³] → all degrees 2 to ~3*cp_degree
                           If False: [r1I², r2I², r12²] → even degrees only
        use_manual_symmetrization: Use manual symmetrization (r1,r2) + (r2,r1) / 2 (default: True)
                                   If False: uses symmetric sums (r1²+r2², r1²*r2I², etc.) - higher degree
        use_raw_distances: Use raw Euclidean distances (default: True)
                          If False: uses BH-style scaled distances r/(1+r)
        include_cusp: Add e-e cusp term u_cusp = 0.5 * r̄12 (default: False)
                     Requires use_raw_distances=False (cusp needs BH-style scaling)
        epsilon: Numerical stability constant (default: 1e-8)
        name: Optional name for parameter filtering
    """

    def __init__(
        self,
        mol,
        cp_degree: int = 3,
        cp_rank: int = 4,
        cp_scale_init: float = 1.0,
        learnable_cp_scale: bool = True,
        use_log_scale: bool = False,
        use_cubic_features: bool = False,
        use_manual_symmetrization: bool = True,
        use_raw_distances: bool = False,
        include_cusp: bool = True,
        epsilon: float = 1e-8,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # Validate cusp configuration
        if include_cusp and use_raw_distances:
            raise ValueError(
                "Cusp term requires BH-style scaled distances. "
                "Set use_raw_distances=False when include_cusp=True."
            )

        # Store molecule info
        self.mol = mol
        self.nuclear_pos = jnp.array(mol.atom_coords())
        self.natom = len(self.nuclear_pos)

        # CP network configuration
        self.cp_degree = int(cp_degree)
        self.cp_rank = int(cp_rank)
        self.cp_scale_init = float(cp_scale_init)

        # Feature configuration
        self.use_cubic_features = bool(use_cubic_features)
        self.use_manual_symmetrization = bool(use_manual_symmetrization)
        self.use_raw_distances = bool(use_raw_distances)
        self.include_cusp = bool(include_cusp)

        # Scale configuration
        self.learnable_cp_scale = bool(learnable_cp_scale)
        self.use_log_scale = bool(use_log_scale)

        # Numerical stability
        self.epsilon = float(epsilon)
        
        # Input dimension: [r1², r2², r12²] or [r1², r2², r12², r1³, r2³, r12³]
        self.cp_input_dim = 6 if self.use_cubic_features else 3

    def _safe_norm(self, x):
        """Compute norm with epsilon to prevent division by zero."""
        return jnp.sqrt(jnp.sum(x * x, axis=-1) + self.epsilon)

    # ========== BH-style scaled distance helpers (copied from bh.py) ==========

    def _scaled_r_en(self, r_electron, r_nuclear):
        """Compute scaled electron-nuclear distance: r/(1+r).

        Args:
            r_electron: Electron position, shape (3,)
            r_nuclear: Nuclear position, shape (3,)

        Returns:
            Scaled distance (scalar)
        """
        r = self._safe_norm(r_electron - r_nuclear)
        return r / (1.0 + r)

    def _scaled_r_ee(self, r1, r2):
        """Compute scaled electron-electron distance: r/(1+r).

        Args:
            r1: First electron position, shape (3,)
            r2: Second electron position, shape (3,)

        Returns:
            Scaled distance (scalar)
        """
        r = self._safe_norm(r1 - r2)
        return r / (1.0 + r)

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

    def init_params(self, **kwargs) -> Dict[str, Any]:
        """Initialize CP parameters.

        Returns dict with:
            - CP params: 'cp_U', 'cp_C', 'cp_bias'
            - CP scale: 'cp_scale' or 'cp_scale_log' (if learnable)
            - Distance scaling (if use_raw_distances=False): 'b_raw', 'd_raw'
        """
        key = kwargs.get("key", random.PRNGKey(0))
        init_scale = float(kwargs.get("cp_init_scale", 1e-2))

        params = {}
        
        # ===== CP network parameters =====
        # U matrices: (natom, degree, input_dim, rank)
        key, *subkeys = random.split(key, self.natom + 1)
        U_list = []
        for nuc_idx in range(self.natom):
            # For this nucleus, create degree matrices
            U_nuc = []
            for _ in range(self.cp_degree):
                U_deg = init_scale * random.normal(
                    subkeys[nuc_idx],
                    (self.cp_input_dim, self.cp_rank)
                )
                U_nuc.append(U_deg)
            U_list.append(jnp.stack(U_nuc, axis=0))  # (degree, input_dim, rank)
        params['cp_U'] = jnp.stack(U_list, axis=0)  # (natom, degree, input_dim, rank)

        # C matrices: (natom, rank)
        key, subkey = random.split(key)
        params['cp_C'] = init_scale * random.normal(subkey, (self.natom, self.cp_rank))

        # Bias per nucleus: (natom,)
        params['cp_bias'] = jnp.zeros(self.natom)

        # Global scale for CP contribution
        if self.learnable_cp_scale:
            if self.use_log_scale:
                # Store log(scale) for more stable optimization
                params['cp_scale_log'] = jnp.array(jnp.log(self.cp_scale_init))
            else:
                # Store scale directly
                params['cp_scale'] = jnp.array(self.cp_scale_init)
        else:
            # Fixed scale - store as constant (will be excluded from gradients in _compute)
            params['_cp_scale_fixed'] = float(self.cp_scale_init)

        # Note: BH-style scaling r/(1+r) is fixed - no learnable b, d parameters needed

        return params

    def _get_cp_scale(self, params: Dict[str, Any]) -> float:
        """Get CP scale parameter (handles learnable vs fixed, log vs direct)."""
        if self.learnable_cp_scale:
            if self.use_log_scale:
                # Exponential of log(scale) to get actual scale
                return jnp.exp(params['cp_scale_log'])
            else:
                # Direct scale parameter
                return params['cp_scale']
        else:
            # Fixed scale - use constant value with stop_gradient to prevent optimization
            return jax.lax.stop_gradient(jnp.array(params['_cp_scale_fixed']))

    def _cp_recursion(self, x: jnp.ndarray, U: jnp.ndarray, C: jnp.ndarray,
                     bias: float, degree: int) -> float:
        """CP polynomial recursion: out = U1(x); for i=2..D: out = Ui(x)*out + out.

        Uses jax.lax.scan for efficient computation.

        Args:
            x: Input features, shape (input_dim,)
            U: Factor matrices, shape (degree, input_dim, rank)
            C: Projection vector, shape (rank,)
            bias: Scalar bias
            degree: Polynomial degree (for loop bound)

        Returns:
            Scalar output
        """
        # First transformation: out = x @ U[0]
        # x: (input_dim,), U[0]: (input_dim, rank) -> out: (rank,)
        out = x @ U[0]  # (rank,)

        # Polynomial recursion using scan
        def scan_fn(out_carry, U_i):
            """Scan function for CP recursion."""
            ui_x = x @ U_i  # (rank,)
            new_out = ui_x * out_carry + out_carry  # Element-wise multiply and add
            return new_out, None

        # Scan over U[1:degree]
        out, _ = jax.lax.scan(scan_fn, out, U[1:degree])

        # Final projection
        result = jnp.dot(out, C) + bias
        return result

    def _cp_recursion_gradient(self, x: jnp.ndarray, U: jnp.ndarray, C: jnp.ndarray,
                               bias: float, degree: int):
        """CP recursion with gradient only (no Hessian) - efficient for XTC.

        Based on the recursive formulas from "Deep Polynomial Neural Networks" (arXiv:2006.13026v2).

        Recursive formulas:
        - State: x_n = (U[n]^T z + 1) ⊙ x_{n-1}
        - Gradient: ∂x_n/∂z_d = (u[n,d] ⊙ x_{n-1}) + (U[n]^T z + 1) ⊙ ∂x_{n-1}/∂z_d

        Args:
            x: Input features, shape (input_dim,)
            U: Factor matrices, shape (degree, input_dim, rank)
            C: Projection vector, shape (rank,)
            bias: Scalar bias
            degree: Polynomial degree

        Returns:
            Tuple of (value, gradient) where:
            - value: scalar output G(x) = C·x_N + bias
            - gradient: ∂G/∂x, shape (input_dim,)
        """
        input_dim = x.shape[0]
        rank = U.shape[2]

        # Initialize state and gradient for n=1
        # x_1 = U[0]^T @ x
        x_state = U[0].T @ x  # (rank,)

        # ∂x_1/∂z_d = u[1,d] (d-th row of U[0])
        dx_state = U[0]  # (input_dim, rank) - each row is ∂x_1/∂z_d

        # Recursive computation for n = 2, ..., degree
        def scan_fn(carry, U_i):
            """Scan function for CP recursion with gradient only."""
            x_prev, dx_prev = carry  # Only 2 variables (no Hessian!)

            # Compute U_i^T @ x + 1
            v = U_i.T @ x + 1.0  # (rank,)

            # State update: x_n = v ⊙ x_{n-1}
            x_new = v * x_prev  # (rank,)

            # Gradient update: ∂x_n/∂z_d = (u[n,d] ⊙ x_{n-1}) + v ⊙ ∂x_{n-1}/∂z_d
            # U_i: (input_dim, rank), U_i[d, :] is u[n,d]
            dx_new = U_i * x_prev[None, :] + v[None, :] * dx_prev  # (input_dim, rank)

            return (x_new, dx_new), None  # Only 2 return values

        # Scan over U[1:degree]
        if degree > 1:
            (x_state, dx_state), _ = jax.lax.scan(
                scan_fn, (x_state, dx_state), U[1:degree]
            )

        # Final output: G(x) = C · x_N + bias
        value = jnp.dot(C, x_state) + bias

        # ∂G/∂z_d = C · ∂x_N/∂z_d
        gradient = dx_state @ C  # (input_dim,)

        return value, gradient

    def _compute_distances(self, r1: jnp.ndarray, r2: jnp.ndarray,
                          nuclear_pos_I: jnp.ndarray):
        """Compute distances (raw or BH-style scaled) for one nucleus.

        Args:
            r1: First electron position, shape (3,)
            r2: Second electron position, shape (3,)
            nuclear_pos_I: Nuclear position, shape (3,)

        Returns:
            Tuple of (r1I, r2I, r12) - scalar distances
        """
        if self.use_raw_distances:
            # Raw Euclidean distances
            r1I = self._safe_norm(r1 - nuclear_pos_I)
            r2I = self._safe_norm(r2 - nuclear_pos_I)
            r12 = self._safe_norm(r1 - r2)
        else:
            # BH-style scaled distances: r/(1+r) - no learnable parameters
            r1I = self._scaled_r_en(r1, nuclear_pos_I)
            r2I = self._scaled_r_en(r2, nuclear_pos_I)
            r12 = self._scaled_r_ee(r1, r2)

        return r1I, r2I, r12

    def _compute_cp_features(self, r1I: float, r2I: float, r12: float) -> jnp.ndarray:
        """Compute CP input features from distances.

        Args:
            r1I: Electron-nucleus distance for electron 1
            r2I: Electron-nucleus distance for electron 2
            r12: Electron-electron distance

        Returns:
            Features array: [r1I², r2I², r12²] or [r1I², r2I², r12², r1I³, r2I³, r12³]
        """
        if self.use_cubic_features:
            return jnp.array([r1I**2, r2I**2, r12**2, r1I**3, r2I**3, r12**3])
        else:
            return jnp.array([r1I**2, r2I**2, r12**2])

    def _compute_cp_features_symmetric(self, r1I: float, r2I: float, r12: float) -> jnp.ndarray:
        """Compute symmetric CP features (alternative to manual symmetrization).

        Uses symmetric combinations to ensure electron exchange symmetry.

        Args:
            r1I: Electron-nucleus distance for electron 1
            r2I: Electron-nucleus distance for electron 2
            r12: Electron-electron distance

        Returns:
            Symmetric features: [r1I²+r2I², r1I²*r2I², r12², ...]
        """
        if self.use_cubic_features:
            return jnp.array([r1I**2 + r2I**2, r1I**2 * r2I**2, r12**2,
                             r1I**3 + r2I**3, r1I**3 * r2I**3, r12**3])
        else:
            return jnp.array([r1I**2 + r2I**2, r1I**2 * r2I**2, r12**2])

    @partial(jax.jit, static_argnums=(0,))
    def _compute(self, r1: jnp.ndarray, r2: jnp.ndarray, params: Dict[str, Any]) -> jnp.ndarray:
        """Compute u = cp_scale * Σ_I u_cp^I.

        Uses jax.lax.scan for efficient looping over nuclei.

        Architecture:
        - Single network with [r1², r2², r12²] or [r1², r2², r12², r1³, r2³, r12³]
        - Manual symmetrization: u = (CP(r1,r2) + CP(r2,r1)) / 2 (default)
        - Or symmetric sums: [r1²+r2², r1²*r2², r12²] (if use_manual_symmetrization=False)

        Args:
            r1: First electron position (3,) or (1, 3)
            r2: Second electron position (3,) or (1, 3)
            params: Parameter dictionary

        Returns:
            Scalar Jastrow exponent
        """
        # Flatten inputs
        r1_flat = jnp.reshape(r1, (3,))
        r2_flat = jnp.reshape(r2, (3,))

        # Compute CP contribution using scan over nuclei
        def cp_nucleus_scan_fn(carry, nucleus_data):
            """Scan function for CP over nuclei."""
            u_cp_total = carry
            nuclear_pos_I, cp_U_I, cp_C_I, cp_bias_I = nucleus_data

            # Compute distances (raw or BH-style scaled)
            r1I, r2I, r12 = self._compute_distances(r1_flat, r2_flat, nuclear_pos_I)

            # Build features
            if self.use_manual_symmetrization:
                # Compute for both electron orderings
                if self.use_cubic_features:
                    x_12 = jnp.array([r1I**2, r2I**2, r12**2, r1I**3, r2I**3, r12**3])
                    x_21 = jnp.array([r2I**2, r1I**2, r12**2, r2I**3, r1I**3, r12**3])
                else:
                    x_12 = jnp.array([r1I**2, r2I**2, r12**2])
                    x_21 = jnp.array([r2I**2, r1I**2, r12**2])

                # Evaluate both orderings and average for guaranteed symmetry
                u_12 = self._cp_recursion(x_12, cp_U_I, cp_C_I, cp_bias_I, self.cp_degree)
                u_21 = self._cp_recursion(x_21, cp_U_I, cp_C_I, cp_bias_I, self.cp_degree)
                u_cp_I = (u_12 + u_21) / 2
            else:
                # Use symmetric sums (higher degree but simpler)
                x = self._compute_cp_features_symmetric(r1I, r2I, r12)
                u_cp_I = self._cp_recursion(x, cp_U_I, cp_C_I, cp_bias_I, self.cp_degree)

            u_cp_total += u_cp_I
            return u_cp_total, None

        # Prepare nucleus data for scan
        nucleus_data = (
            self.nuclear_pos,
            params['cp_U'],
            params['cp_C'],
            params['cp_bias'],
        )

        # Scan over nuclei
        u_cp_total, _ = jax.lax.scan(cp_nucleus_scan_fn, 0.0, nucleus_data)

        # Add cusp term if enabled
        if self.include_cusp:
            # Compute scaled e-e distance
            r12_scaled = self._scaled_r_ee(r1_flat, r2_flat)
            # Cusp contribution: u_cusp = 0.5 * r̄12 (coefficient 0.5 is fixed)
            u_cusp = 0.5 * r12_scaled
            u_total = u_cusp + u_cp_total
        else:
            u_total = u_cp_total

        # Get CP scale
        cp_scale = self._get_cp_scale(params)

        # Return scaled contribution
        return cp_scale * u_total

    def get_cp_param_count(self) -> int:
        """Get number of CP parameters.

        Returns:
            Total number of CP parameters
        """
        # Single network with input_dim features
        # U: (natom, degree, input_dim, rank)
        # C: (natom, rank)
        # bias: (natom,)
        # scale: 1 (if learnable)
        # Note: BH-style scaling r/(1+r) has no learnable parameters
        n_U = self.natom * self.cp_degree * self.cp_input_dim * self.cp_rank
        n_C = self.natom * self.cp_rank
        n_bias = self.natom
        n_scale = 1 if self.learnable_cp_scale else 0
        return n_U + n_C + n_bias + n_scale

    # ========== CP feature gradient helper methods ==========

    def _grad_cp_features_wrt_r1I(self, r1I: float, r2I: float,
                                   grad_features: jnp.ndarray,
                                   is_x12: bool = True) -> float:
        """Chain rule: convert ∂u/∂features to ∂u/∂r1I.

        Args:
            r1I: Electron-nucleus distance for electron 1
            r2I: Electron-nucleus distance for electron 2
            grad_features: ∂u/∂features from CP recursion, shape (input_dim,)
            is_x12: If True, features are [r1I², r2I², ...]. If False, [r2I², r1I², ...]

        Returns:
            ∂u/∂r1I (scalar)
        """
        if self.use_manual_symmetrization:
            # For x_12 = [r1I², r2I², r12², r1I³, r2I³, r12³]: r1I² at index 0, r1I³ at index 3
            # For x_21 = [r2I², r1I², r12², r2I³, r1I³, r12³]: r1I² at index 1, r1I³ at index 4
            if is_x12:
                if self.use_cubic_features:
                    du_dr1I_sq = grad_features[0]
                    du_dr1I_cu = grad_features[3]
                else:
                    du_dr1I_sq = grad_features[0]
                    du_dr1I_cu = 0.0
            else:  # x_21 ordering
                if self.use_cubic_features:
                    du_dr1I_sq = grad_features[1]
                    du_dr1I_cu = grad_features[4]
                else:
                    du_dr1I_sq = grad_features[1]
                    du_dr1I_cu = 0.0
        else:
            # Symmetric features: [r1I²+r2I², r1I²*r2I², r12², ...]
            # ∂(r1I²+r2I²)/∂r1I = 2*r1I
            # ∂(r1I²*r2I²)/∂r1I = 2*r1I*r2I²
            du_dr1I_sq = grad_features[0] + grad_features[1] * 2.0 * r2I**2
            if self.use_cubic_features:
                # ∂(r1I³+r2I³)/∂r1I = 3*r1I²
                # ∂(r1I³*r2I³)/∂r1I = 3*r1I²*r2I³
                du_dr1I_cu = grad_features[3] + grad_features[4] * 3.0 * r2I**3
            else:
                du_dr1I_cu = 0.0

        # Chain rule: ∂u/∂r1I = (∂u/∂(r1I²)) * (∂(r1I²)/∂r1I) + (∂u/∂(r1I³)) * (∂(r1I³)/∂r1I)
        # ∂(r1I²)/∂r1I = 2*r1I, ∂(r1I³)/∂r1I = 3*r1I²
        return du_dr1I_sq * 2.0 * r1I + du_dr1I_cu * 3.0 * r1I**2

    def _grad_cp_features_wrt_r2I(self, r1I: float, r2I: float,
                                   grad_features: jnp.ndarray,
                                   is_x12: bool = True) -> float:
        """Chain rule: convert ∂u/∂features to ∂u/∂r2I.

        Args:
            r1I: Electron-nucleus distance for electron 1
            r2I: Electron-nucleus distance for electron 2
            grad_features: ∂u/∂features from CP recursion, shape (input_dim,)
            is_x12: If True, features are [r1I², r2I², ...]. If False, [r2I², r1I², ...]

        Returns:
            ∂u/∂r2I (scalar)
        """
        if self.use_manual_symmetrization:
            # For x_12 = [r1I², r2I², r12², r1I³, r2I³, r12³]: r2I² at index 1, r2I³ at index 4
            # For x_21 = [r2I², r1I², r12², r2I³, r1I³, r12³]: r2I² at index 0, r2I³ at index 3
            if is_x12:
                if self.use_cubic_features:
                    du_dr2I_sq = grad_features[1]
                    du_dr2I_cu = grad_features[4]
                else:
                    du_dr2I_sq = grad_features[1]
                    du_dr2I_cu = 0.0
            else:  # x_21 ordering
                if self.use_cubic_features:
                    du_dr2I_sq = grad_features[0]
                    du_dr2I_cu = grad_features[3]
                else:
                    du_dr2I_sq = grad_features[0]
                    du_dr2I_cu = 0.0
        else:
            # Symmetric features: [r1I²+r2I², r1I²*r2I², r12², ...]
            # ∂(r1I²+r2I²)/∂r2I = 2*r2I
            # ∂(r1I²*r2I²)/∂r2I = 2*r2I*r1I²
            du_dr2I_sq = grad_features[0] + grad_features[1] * 2.0 * r1I**2
            if self.use_cubic_features:
                # ∂(r1I³+r2I³)/∂r2I = 3*r2I²
                # ∂(r1I³*r2I³)/∂r2I = 3*r2I²*r1I³
                du_dr2I_cu = grad_features[3] + grad_features[4] * 3.0 * r1I**3
            else:
                du_dr2I_cu = 0.0

        # Chain rule: ∂(r2I²)/∂r2I = 2*r2I, ∂(r2I³)/∂r2I = 3*r2I²
        return du_dr2I_sq * 2.0 * r2I + du_dr2I_cu * 3.0 * r2I**2

    def _grad_cp_features_wrt_r12(self, r12: float, grad_features: jnp.ndarray) -> float:
        """Chain rule: convert ∂u/∂features to ∂u/∂r12.

        Args:
            r12: Electron-electron distance
            grad_features: ∂u/∂features from CP recursion, shape (input_dim,)

        Returns:
            ∂u/∂r12 (scalar)
        """
        # r12 appears at index 2 in both manual and symmetric features
        if self.use_cubic_features:
            du_dr12_sq = grad_features[2]
            du_dr12_cu = grad_features[5]
        else:
            du_dr12_sq = grad_features[2]
            du_dr12_cu = 0.0

        # Chain rule: ∂(r12²)/∂r12 = 2*r12, ∂(r12³)/∂r12 = 3*r12²
        return du_dr12_sq * 2.0 * r12 + du_dr12_cu * 3.0 * r12**2

    # ========== Analytical gradient methods ==========

    def _grad_cp_analytical(self, r1: jnp.ndarray, r2: jnp.ndarray,
                           params: Dict[str, Any]) -> jnp.ndarray:
        """Compute analytical gradient of CP contribution w.r.t r1.

        Uses chain rule: ∇_r1 u_CP = Σ_I [(∂u_CP/∂r1I)·∇r1I + (∂u_CP/∂r12)·∇r12]

        Handles both raw and BH-style scaled distances.

        Args:
            r1: First electron position, shape (3,)
            r2: Second electron position, shape (3,)
            params: Jastrow parameters

        Returns:
            Gradient vector, shape (3,)
        """
        def nucleus_scan_fn(carry, nucleus_data):
            """Scan function for CP gradient over nuclei."""
            grad_total = carry
            nuclear_pos_I, cp_U_I, cp_C_I, cp_bias_I = nucleus_data

            # Compute distances (raw or BH-style scaled)
            r1I, r2I, r12 = self._compute_distances(r1, r2, nuclear_pos_I)

            # Compute distance gradients
            if self.use_raw_distances:
                # Raw distances: gradient is just the unit vector
                r1I_vec = r1 - nuclear_pos_I
                r12_vec = r1 - r2
                grad_r1I = r1I_vec / (r1I + self.epsilon)
                grad_r12 = r12_vec / (r12 + self.epsilon)
            else:
                # BH-style scaled distances: use helper methods
                grad_r1I = self._grad_scaled_r_en(r1, nuclear_pos_I)
                grad_r12 = self._grad_scaled_r_ee(r1, r2)

            # Build CP features and compute derivatives
            if self.use_manual_symmetrization:
                # Compute for both electron orderings
                x_12 = self._compute_cp_features(r1I, r2I, r12)
                x_21 = self._compute_cp_features(r2I, r1I, r12)  # Swapped!

                # Use gradient-only recursion
                _, grad_x_12 = self._cp_recursion_gradient(
                    x_12, cp_U_I, cp_C_I, cp_bias_I, self.cp_degree
                )
                _, grad_x_21 = self._cp_recursion_gradient(
                    x_21, cp_U_I, cp_C_I, cp_bias_I, self.cp_degree
                )

                # Average gradients for symmetrization
                du_dr1I_12 = self._grad_cp_features_wrt_r1I(r1I, r2I, grad_x_12, is_x12=True)
                du_dr1I_21 = self._grad_cp_features_wrt_r1I(r1I, r2I, grad_x_21, is_x12=False)
                du_dr1I = (du_dr1I_12 + du_dr1I_21) / 2.0

                du_dr12_12 = self._grad_cp_features_wrt_r12(r12, grad_x_12)
                du_dr12_21 = self._grad_cp_features_wrt_r12(r12, grad_x_21)
                du_dr12 = (du_dr12_12 + du_dr12_21) / 2.0
            else:
                # Use symmetric features
                x = self._compute_cp_features_symmetric(r1I, r2I, r12)

                # Use gradient-only recursion
                _, grad_x = self._cp_recursion_gradient(
                    x, cp_U_I, cp_C_I, cp_bias_I, self.cp_degree
                )

                # Chain rule: features → distances
                du_dr1I = self._grad_cp_features_wrt_r1I(r1I, r2I, grad_x, is_x12=True)
                du_dr12 = self._grad_cp_features_wrt_r12(r12, grad_x)

            # Chain rule: distances → positions
            # ∇u = (∂u/∂r1I)·∇r1I + (∂u/∂r12)·∇r12
            grad_contrib = du_dr1I * grad_r1I + du_dr12 * grad_r12

            return grad_total + grad_contrib, None

        # Pack nucleus data for scan
        nucleus_data = (
            self.nuclear_pos,
            params['cp_U'],      # (natom, degree, input_dim, rank)
            params['cp_C'],      # (natom, rank)
            params['cp_bias'],   # (natom,)
        )

        # Scan over nuclei to get CP gradient
        grad_cp, _ = jax.lax.scan(nucleus_scan_fn, jnp.zeros(3), nucleus_data)

        # Add cusp gradient if enabled
        if self.include_cusp:
            # ∇u_cusp = 0.5 * ∇r̄12
            # where ∇r̄12 = (r1-r2) / [r12 * (1+r12)²]
            grad_r12_cusp = self._grad_scaled_r_ee(r1, r2)
            grad_cusp = 0.5 * grad_r12_cusp
            grad_total = grad_cp + grad_cusp
        else:
            grad_total = grad_cp

        return grad_total

    @partial(jax.jit, static_argnums=(0,))
    def grad_r(self, r1, r2, params):
        """Compute gradient of u w.r.t r1 using analytical derivatives.

        Works for both raw and BH-style scaled distances.

        Args:
            r1: First electron position, shape (3,) or (1, 3)
            r2: Second electron position, shape (3,) or (1, 3)
            params: Jastrow parameters

        Returns:
            Gradient vector of shape (3,)
        """
        # Store original shape
        original_shape = r1.shape
        r1_flat = jnp.reshape(r1, (3,))
        r2_flat = jnp.reshape(r2, (3,))

        # Use analytical gradient (works for both raw and scaled)
        grad_cp = self._grad_cp_analytical(r1_flat, r2_flat, params)
        cp_scale = self._get_cp_scale(params)
        grad_total = cp_scale * grad_cp

        return jnp.reshape(grad_total, original_shape)

    def _laplacian_cp_analytical(self, r1: jnp.ndarray, r2: jnp.ndarray,
                                 params: Dict[str, Any]) -> float:
        """Compute Laplacian of CP contribution w.r.t r1.

        Uses hybrid approach: analytical gradient + jacfwd for Jacobian.
        This is simpler than full analytical Hessian and still provides good speedup.
        Works for both raw and BH-style scaled distances.

        Args:
            r1: First electron position, shape (3,)
            r2: Second electron position, shape (3,)
            params: Jastrow parameters

        Returns:
            Laplacian (scalar)
        """
        def grad_fn(x):
            return self._grad_cp_analytical(x, r2, params)

        jacobian = jax.jacfwd(grad_fn)(r1)
        return jnp.trace(jacobian)

    @partial(jax.jit, static_argnums=(0,))
    def laplacian_r(self, r1, r2, params):
        """Compute Laplacian of u w.r.t r1.

        Uses hybrid approach (analytical gradient + jacfwd).
        Works for both raw and BH-style scaled distances.

        Args:
            r1: First electron position, shape (3,) or (1, 3)
            r2: Second electron position, shape (3,) or (1, 3)
            params: Jastrow parameters

        Returns:
            Laplacian (scalar)
        """
        # Flatten inputs
        r1_flat = jnp.reshape(r1, (3,))
        r2_flat = jnp.reshape(r2, (3,))

        # Use hybrid analytical Laplacian (works for both raw and scaled)
        lapl_cp = self._laplacian_cp_analytical(r1_flat, r2_flat, params)
        cp_scale = self._get_cp_scale(params)
        return cp_scale * lapl_cp

    @partial(jax.jit, static_argnums=(0,))
    def get_log_grads_r1(self, r1, r2, params):
        """Compute gradient and Laplacian w.r.t r1.

        Args:
            r1: First electron position, shape (3,) or (1, 3)
            r2: Second electron position, shape (3,) or (1, 3)
            params: Jastrow parameters

        Returns:
            Tuple of (gradient, Laplacian)
        """
        grad_u = self.grad_r(r1, r2, params)
        lapl_u = self.laplacian_r(r1, r2, params)
        return grad_u, lapl_u

    @partial(jax.jit, static_argnums=(0,))
    def get_log_grads_r2(self, r1, r2, params):
        """Compute gradient and Laplacian w.r.t r2.

        Uses electron exchange symmetry: swap r1 and r2.

        Args:
            r1: First electron position, shape (3,) or (1, 3)
            r2: Second electron position, shape (3,) or (1, 3)
            params: Jastrow parameters

        Returns:
            Tuple of (gradient, Laplacian)
        """
        # Use symmetry: u(r1, r2) = u(r2, r1)
        # So ∇_r2 u(r1, r2) = ∇_r1 u(r2, r1)
        grad_u = self.grad_r(r2, r1, params)
        lapl_u = self.laplacian_r(r2, r1, params)
        return grad_u, lapl_u

