"""Analytical coordinate gradients and Laplacians for the CP Jastrow."""

from flax import struct
import jax
import jax.numpy as jnp

from .cp import CP


@struct.dataclass
class CPAnalytical(CP):
    """The same CP model with explicit gradient and Laplacian recurrences.

    Forward evaluation, initialization, configuration and parameter gradients
    are inherited unchanged from CP. Construct with ``CPAnalytical.create``;
    parameter dictionaries are interchangeable with an identically configured
    CP. Coordinate derivatives use no autodiff or full Hessian construction.
    Parameter differentiation still uses autodiff. Batch and pair-grid methods
    inherited from Jastrow dispatch to the overrides here.

    Positive epsilon gives exact derivatives of the *smoothed* distance used
    by CP, including coalescence. With epsilon=0, derivatives at exact
    coalescence remain undefined, as in CP.
    """

    def _distance_and_gradient(self, displacement):
        r = jnp.sqrt(jnp.sum(displacement * displacement) + self.epsilon)
        grad = displacement / r
        if self.use_raw_distances:
            return r, grad
        return r / (1.0 + r), grad / (1.0 + r)**2

    def _distance_derivatives(self, displacement):
        """Value, gradient and exact 3D Laplacian of the smoothed distance.

        For r=sqrt(|d|^2+epsilon), |grad r|^2=|d|^2/r^2 and
        lap r=2/r+epsilon/r^3. The epsilon term cannot be dropped near
        coalescence. For a scaled distance f(r), lap f=f''|grad r|^2+f'lap r.
        """
        r = jnp.sqrt(jnp.sum(displacement * displacement) + self.epsilon)
        gradient = displacement / r
        laplacian = (2.0 + self.epsilon / r**2) / r
        if self.use_raw_distances:
            return r, gradient, laplacian
        denominator = 1.0 + r
        lap_scaled = (laplacian / denominator**2
                      - 2.0 * jnp.sum(gradient**2) / denominator**3)
        return r / denominator, gradient / denominator**2, lap_scaled

    def _feature_gradients(self, a, b, c, da, db, dc):
        """Cartesian derivatives in exactly the ordering of CP._features."""
        powers = (2, 3) if self.use_cubic_features else (2,)
        if self.use_manual_symmetrization:
            return jnp.stack([
                p * v**(p-1) * dv
                for p in powers for v, dv in ((a, da), (b, db), (c, dc))
            ])
        # d(a^p b^p) = p*a^(p-1)*b^p*da + p*a^p*b^(p-1)*db.
        # The chain-rule factor p appears once, not twice.
        return jnp.stack([
            dv for p in powers for dv in (
                p * (a**(p-1)*da + b**(p-1)*db),
                p * (a**(p-1)*b**p*da + a**p*b**(p-1)*db),
                p * c**(p-1)*dc,
            )
        ])

    def _feature_laplacians(self, a, b, c, da, db, dc, la, lb, lc):
        """Power and product rules in the same order as CP._features."""
        powers = (2, 3) if self.use_cubic_features else (2,)
        result = []
        for p in powers:
            powered = []
            for value, gradient, laplacian in ((a, da, la), (b, db, lb), (c, dc, lc)):
                powered.append((
                    value**p,
                    p * value**(p-1) * gradient,
                    p * value**(p-1) * laplacian
                    + p * (p-1) * value**(p-2) * jnp.sum(gradient**2),
                ))
            (ap, dap, lap_a), (bp, dbp, lap_b), (_, _, lap_c) = powered
            if self.use_manual_symmetrization:
                result.extend((lap_a, lap_b, lap_c))
            else:
                result.extend((lap_a + lap_b,
                               bp * lap_a + ap * lap_b + 2.0 * jnp.dot(dap, dbp),
                               lap_c))
        return jnp.stack(result)

    def _polynomial_gradient(self, x, dx, factors, projection):
        """Propagate d(state)/dr alongside state, without dividing by state.

        For state' = state * (1 + x @ U),
        dstate' = dstate * (1 + x @ U) + state * d(x @ U).
        This remains valid when either factor vanishes. Bias has no spatial
        derivative but remains part of the inherited forward calculation.
        """
        state = x @ factors[0]
        derivative = factors[0].T @ dx
        for stage in range(1, self.cp_degree):
            multiplier = 1.0 + x @ factors[stage]
            derivative = (derivative * multiplier[:, None]
                          + state[:, None] * (factors[stage].T @ dx))
            state = state * multiplier
        return projection @ derivative

    def _polynomial_derivatives(self, x, dx, lx, factors, projection):
        """Propagate state, gradient and Laplacian through each product.

        lap(state * multiplier) = multiplier * lap(state)
            + state * lap(multiplier) + 2 * grad(state).grad(multiplier).
        All right-hand sides use the previous state. No division is needed,
        including when a polynomial factor is zero.
        """
        state = x @ factors[0]
        gradient = factors[0].T @ dx
        laplacian = lx @ factors[0]
        for stage in range(1, self.cp_degree):
            multiplier = 1.0 + x @ factors[stage]
            dm = factors[stage].T @ dx
            lm = lx @ factors[stage]
            laplacian = (multiplier * laplacian + state * lm
                         + 2.0 * jnp.sum(gradient * dm, axis=-1))
            gradient = gradient * multiplier[:, None] + state[:, None] * dm
            state = state * multiplier
        return projection @ gradient, jnp.dot(projection, laplacian)

    def grad_r(self, r1, r2, params):
        """Analytical gradient of the pair exponent with respect to r1."""
        original_shape = r1.shape
        r1, r2 = jnp.reshape(r1, (3,)), jnp.reshape(r2, (3,))
        c, dc = self._distance_and_gradient(r1 - r2)
        zero = jnp.zeros_like(r1)

        def nucleus_gradient(position, factors, projection):
            a, da = self._distance_and_gradient(r1 - position)
            b = self._distance(r2 - position)
            gradient = self._polynomial_gradient(
                self._features(a, b, c),
                self._feature_gradients(a, b, c, da, zero, dc),
                factors, projection,
            )
            if self.use_manual_symmetrization:
                swapped = self._polynomial_gradient(
                    self._features(b, a, c),
                    self._feature_gradients(b, a, c, zero, da, dc),
                    factors, projection,
                )
                gradient = 0.5 * (gradient + swapped)
            return gradient

        residual_grad = jnp.sum(jax.vmap(nucleus_gradient)(
            self.nuclear_pos, params["cp_U"], params["cp_C"]
        ), axis=0)
        result = self._get_cp_scale(params) * residual_grad
        if self.include_cusp:
            result = result + 0.5 * dc
        return jnp.reshape(result, original_shape)

    def laplacian_r(self, r1, r2, params):
        """Fully analytical coordinate Laplacian of the pair exponent."""
        return self.get_log_grads_r1(r1, r2, params)[1]

    def get_log_grads_r1(self, r1, r2, params):
        """Return the analytical coordinate gradient and Laplacian together."""
        original_shape = r1.shape
        r1, r2 = jnp.reshape(r1, (3,)), jnp.reshape(r2, (3,))
        c, dc, lc = self._distance_derivatives(r1 - r2)
        zero = jnp.zeros_like(r1)

        def nucleus_derivatives(position, factors, projection):
            a, da, la = self._distance_derivatives(r1 - position)
            b = self._distance(r2 - position)
            gradient, laplacian = self._polynomial_derivatives(
                self._features(a, b, c),
                self._feature_gradients(a, b, c, da, zero, dc),
                self._feature_laplacians(a, b, c, da, zero, dc, la, 0.0, lc),
                factors, projection,
            )
            if self.use_manual_symmetrization:
                other_g, other_l = self._polynomial_derivatives(
                    self._features(b, a, c),
                    self._feature_gradients(b, a, c, zero, da, dc),
                    self._feature_laplacians(b, a, c, zero, da, dc, 0.0, la, lc),
                    factors, projection,
                )
                gradient = 0.5 * (gradient + other_g)
                laplacian = 0.5 * (laplacian + other_l)
            return gradient, laplacian

        gradients, laplacians = jax.vmap(nucleus_derivatives)(
            self.nuclear_pos, params["cp_U"], params["cp_C"]
        )
        scale = self._get_cp_scale(params)
        gradient = scale * jnp.sum(gradients, axis=0)
        laplacian = scale * jnp.sum(laplacians)
        if self.include_cusp:
            gradient = gradient + 0.5 * dc
            laplacian = laplacian + 0.5 * lc
        return jnp.reshape(gradient, original_shape), laplacian

    def get_log_grads_r2(self, r1, r2, params):
        """Electron exchange symmetry supplies the second-electron derivative."""
        return self.get_log_grads_r1(r2, r1, params)
