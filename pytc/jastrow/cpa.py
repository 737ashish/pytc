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
    Parameter differentiation still uses autodiff. ``use_shared_pair_grid``
    selects shared-intermediate batching for both rectangular gradient batches
    and VMC pair grids. Disable it to use the generic per-pair reference path.
    Cached intermediates are local to each call, never persisted across updates.

    Positive epsilon gives exact derivatives of the *smoothed* distance used
    by CP, including coalescence. With epsilon=0, derivatives at exact
    coalescence remain undefined, as in CP.
    """

    use_shared_pair_grid: bool = struct.field(pytree_node=False, default=False)

    @classmethod
    def create(cls, mol, *args, use_shared_pair_grid=False, **kwargs):
        """Create CPA; optionally reuse distance tables across batched pairs.

        Other arguments and all parameters have the same meaning as CP.create.
        The generic path remains available for benchmarking and validation.
        """
        return super().create(mol, *args, **kwargs).replace(
            use_shared_pair_grid=bool(use_shared_pair_grid))

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

    def _distance_power_tables(self, displacement, with_laplacian):
        """Compute powers and derivatives once per electron/nucleus or pair."""
        shape = displacement.shape[:-1]
        flat = displacement.reshape((-1, 3))
        if with_laplacian:
            value, gradient, laplacian = jax.vmap(self._distance_derivatives)(flat)
        else:
            value, gradient = jax.vmap(self._distance_and_gradient)(flat)
            laplacian = jnp.zeros_like(value)
        powers = (2, 3) if self.use_cubic_features else (2,)
        values = jnp.stack([value**p for p in powers], axis=-1)
        gradients = jnp.stack([p*value[:, None]**(p-1)*gradient for p in powers], axis=-2)
        if with_laplacian:
            laps = jnp.stack([
                p*value**(p-1)*laplacian
                + p*(p-1)*value**(p-2)*jnp.sum(gradient**2, axis=-1)
                for p in powers
            ], axis=-1)
        else:
            laps = jnp.zeros_like(values)
        return (values.reshape(shape+(len(powers),)),
                gradients.reshape(shape+(len(powers), 3)),
                laps.reshape(shape+(len(powers),)))

    def _shared_pair_derivatives(self, left, right, params, with_laplacian, diagonal):
        """Reuse O((A+B)M) nuclear and O(AB) pair tables across the polynomial.

        Scan over nuclei to avoid materializing an (A,B,M,rank) forward state.
        Polynomial evaluation is still pair- and nucleus-dependent. In VMC,
        match Jastrow's index-based displacement of the ignored diagonal;
        distinct electrons at coincident coordinates are not masked.
        """
        a, da, la = self._distance_power_tables(
            left[:, None, :] - self.nuclear_pos[None, :, :], with_laplacian)
        powers = (2, 3) if self.use_cubic_features else (2,)

        def right_powers(points):
            distance = self._distance(points[:, None, :] - self.nuclear_pos[None, :, :])
            return jnp.stack([distance**p for p in powers], axis=-1)

        b = right_powers(right)
        if diagonal:
            indices = jnp.arange(left.shape[0])
            mask = indices[:, None] == indices[None, :]
            shifted_b = right_powers(right + 1.0)
            pair_difference = left[:, None, :] - jnp.where(
                mask[..., None], right[None, :, :] + 1.0, right[None, :, :])
        else:
            shifted_b = b
            pair_difference = left[:, None, :] - right[None, :, :]
        c, dc, lc = self._distance_power_tables(pair_difference, with_laplacian)
        shape = (left.shape[0], right.shape[0])
        feature_shape = shape + (self.cp_input_dim,)

        def accumulate(carry, data):
            ap, dap, lap, bp, sbp, factors, projection = data
            ap = jnp.broadcast_to(ap[:, None, :], c.shape)
            dap = jnp.broadcast_to(dap[:, None, :, :], dc.shape)
            lap = jnp.broadcast_to(lap[:, None, :], lc.shape)
            bp = jnp.broadcast_to(bp[None, :, :], c.shape)
            if diagonal:
                bp = jnp.where(mask[..., None], sbp[None, :, :], bp)
            zero = jnp.zeros_like(dc)
            if self.use_manual_symmetrization:
                x = jnp.stack((ap, bp, c), axis=-1).reshape(feature_shape)
                dx = jnp.stack((dap, zero, dc), axis=-2).reshape(feature_shape+(3,))
                lx = jnp.stack((lap, jnp.zeros_like(lc), lc), axis=-1).reshape(feature_shape)
            else:
                x = jnp.stack((ap+bp, ap*bp, c), axis=-1).reshape(feature_shape)
                dx = jnp.stack((dap, dap*bp[..., None], dc), axis=-2).reshape(feature_shape+(3,))
                lx = jnp.stack((lap, lap*bp, lc), axis=-1).reshape(feature_shape)

            def evaluate(x, dx, lx):
                if with_laplacian:
                    return self._polynomial_derivatives(x, dx, lx, factors, projection)
                return self._polynomial_gradient(x, dx, factors, projection), jnp.zeros((), dtype=x.dtype)

            evaluate_grid = jax.vmap(jax.vmap(evaluate))
            gradient, laplacian = evaluate_grid(x, dx, lx)
            if self.use_manual_symmetrization:
                # Swap the electron-nucleus feature slots in each power group.
                permutation = jnp.array([1, 0, 2, 4, 3, 5] if self.use_cubic_features else [1, 0, 2])
                other_g, other_l = evaluate_grid(x[..., permutation], dx[..., permutation, :], lx[..., permutation])
                gradient, laplacian = 0.5*(gradient+other_g), 0.5*(laplacian+other_l)
            return (carry[0]+gradient, carry[1]+laplacian), None

        data = tuple(jnp.moveaxis(t, 1, 0) for t in (a, da, la, b, shifted_b))
        data += (params["cp_U"], params["cp_C"])
        dtype = jnp.result_type(a, params["cp_U"], params["cp_C"], self._get_cp_scale(params))
        initial = (jnp.zeros(shape+(3,), dtype=dtype), jnp.zeros(shape, dtype=dtype))
        # Recompute per-nucleus polynomial intermediates during reverse-mode
        # differentiation instead of retaining a full pair-grid tape per atom.
        (gradient, laplacian), _ = jax.lax.scan(
            jax.checkpoint(accumulate), initial, data
        )
        scale = self._get_cp_scale(params)
        gradient, laplacian = scale*gradient, scale*laplacian
        if self.include_cusp:
            # The cusp uses the distance itself, not its squared/cubic powers.
            flat = pair_difference.reshape((-1, 3))
            if with_laplacian:
                _, dg, dl = jax.vmap(self._distance_derivatives)(flat)
                laplacian = laplacian + 0.5*dl.reshape(shape)
            else:
                _, dg = jax.vmap(self._distance_and_gradient)(flat)
            gradient = gradient + 0.5*dg.reshape(shape+(3,))
        return gradient, laplacian

    def grad_r_batch(self, r1_batch, r2_batch, params):
        """Rectangular TC/XTC gradient grid; optionally reuse nuclear tables."""
        if not self.use_shared_pair_grid:
            return super().grad_r_batch(r1_batch, r2_batch, params)
        return self._shared_pair_derivatives(r1_batch, r2_batch, params, False, False)[0]

    def get_pair_grid_grad_lap(self, elec_coords, params):
        """VMC grid with the same diagonal convention as the generic path."""
        if not self.use_shared_pair_grid:
            return super().get_pair_grid_grad_lap(elec_coords, params)
        return self._shared_pair_derivatives(elec_coords, elec_coords, params, True, True)

    def get_log_grads_r2(self, r1, r2, params):
        """Electron exchange symmetry supplies the second-electron derivative."""
        return self.get_log_grads_r1(r2, r1, params)
