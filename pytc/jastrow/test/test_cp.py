"""Baseline CP contracts; compatible with upstream unittest discovery."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto

from pytc.jastrow import CP, Jastrow, BoysHandy, BoysHandyAnalytical, CompositeJastrow
from pytc.vmc.hamiltonian import compute_jastrow_terms

jax.config.update("jax_enable_x64", True)
R1 = jnp.array([0.3, -0.2, 0.6])
R2 = jnp.array([-0.5, 0.4, -0.3])


def mol():
    return gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="bohr", verbose=0)


def reference(model, r1, r2, params):
    """Independent product form of the CP recurrence (no production helpers)."""
    def distance(a, b):
        r = jnp.sqrt(jnp.sum((a - b)**2) + model.epsilon)
        return r if model.use_raw_distances else r / (1 + r)

    def network(a, b, c, atom):
        powers = (2, 3) if model.use_cubic_features else (2,)
        if model.use_manual_symmetrization:
            x = jnp.array([v**p for p in powers for v in (a, b, c)])
        else:
            x = jnp.array([v for p in powers for v in (a**p+b**p, a**p*b**p, c**p)])
        projections = jnp.einsum("d,kdr->kr", x, params["cp_U"][atom])
        state = projections[0] * jnp.prod(1 + projections[1:], axis=0)
        return jnp.dot(state, params["cp_C"][atom]) + params["cp_bias"][atom]

    c = distance(r1, r2)
    total = 0.0
    for atom, pos in enumerate(model.nuclear_pos):
        a, b = distance(r1, pos), distance(r2, pos)
        value = network(a, b, c, atom)
        if model.use_manual_symmetrization:
            value = (value + network(b, a, c, atom)) / 2
        total += value
    scale = model.cp_scale_init
    if model.learnable_cp_scale:
        scale = jnp.exp(params["cp_scale_log"]) if model.use_log_scale else params["cp_scale"]
    return scale * total + (0.5 * c if model.include_cusp else 0.0)


def varied_params(model):
    # Nonidentical stages and sizable residuals detect ordering/derivative errors.
    p = model.init_params(key=jax.random.PRNGKey(17), cp_init_scale=0.2)
    p["cp_U"] = jax.random.normal(jax.random.PRNGKey(31), p["cp_U"].shape) * 0.2
    p["cp_bias"] = jnp.arange(model.natom) * 0.03
    return p


def _check_values_symmetry_and_derivatives(mol, cubic, manual, raw, degree):
    cp = CP.create(mol, cp_degree=degree, use_cubic_features=cubic,
                   use_manual_symmetrization=manual, use_raw_distances=raw,
                   include_cusp=not raw)
    p = varied_params(cp)
    expected = reference(cp, R1, R2, p)
    np.testing.assert_allclose(cp._compute(R1, R2, p), expected, atol=1e-13)
    np.testing.assert_allclose(cp._compute(R2, R1, p), expected, atol=1e-13)
    np.testing.assert_allclose(cp(R1, R2, p), jnp.exp(expected), atol=1e-13)
    for index, method in [(0, cp.get_log_grads_r1), (1, cp.get_log_grads_r2)]:
        f = (lambda x: reference(cp, x, R2, p)) if index == 0 else (lambda x: reference(cp, R1, x, p))
        r = R1 if index == 0 else R2
        grad, lap = method(R1, R2, p)
        np.testing.assert_allclose(grad, jax.grad(f)(r), rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(lap, jnp.trace(jax.hessian(f)(r)), rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(cp.grad_r(R1, R2, p), cp.get_log_grads_r1(R1, R2, p)[0], atol=1e-10)
    np.testing.assert_allclose(cp.laplacian_r(R1, R2, p), cp.get_log_grads_r1(R1, R2, p)[1], atol=1e-10)


def _check_initialization_and_scale(mol, learnable, log):
    cp = CP.create(mol, cp_scale_init=0.7, learnable_cp_scale=learnable, use_log_scale=log)
    p = cp.init_params(key=jax.random.PRNGKey(7))
    assert p["cp_U"].shape == (2, 3, 3, 4)
    assert sum(v.size for v in p.values()) == cp.get_cp_param_count()
    assert "_cp_scale_fixed" not in p
    np.testing.assert_array_equal(p["cp_U"][:, 0], p["cp_U"][:, 1])
    for key, value in cp.init_params(key=jax.random.PRNGKey(7)).items():
        np.testing.assert_array_equal(value, p[key])
    np.testing.assert_allclose(cp._compute(R1, R2, p), reference(cp, R1, R2, p))
    zero = {**p, "cp_C": jnp.zeros_like(p["cp_C"]), "cp_bias": jnp.zeros_like(p["cp_bias"])}
    r = jnp.sqrt(jnp.sum((R1-R2)**2) + cp.epsilon)
    np.testing.assert_allclose(cp._compute(R1, R2, zero), 0.5*r/(1+r))
    if learnable:
        key = "cp_scale_log" if log else "cp_scale"
        np.testing.assert_allclose(cp.grad_params(R1, R2, zero)[key], 0, atol=1e-15)


def _check_dataclass_and_jit(mol):
    cp = CP.create(mol)
    assert isinstance(cp, Jastrow)
    with unittest.TestCase().assertRaises(FrozenInstanceError):
        cp.epsilon = 1.0
    p = cp.init_params()
    evaluate = jax.jit(lambda model, params: model._compute(R1, R2, params))
    np.testing.assert_allclose(evaluate(cp, p), cp._compute(R1, R2, p))
    shifted = cp.replace(nuclear_pos=cp.nuclear_pos + 0.3)
    np.testing.assert_allclose(evaluate(shifted, p), shifted._compute(R1, R2, p))
    assert not np.isclose(float(evaluate(cp, p)), float(evaluate(shifted, p)), rtol=1e-9)


def _check_batch_and_pair_grid(mol):
    cp = CP.create(mol)
    p = varied_params(cp)
    coords = jnp.stack([R1, R2, R1 + 0.4])
    batch = jax.jit(cp.grad_r_batch)(coords, coords[:2], p)
    assert batch.shape == (3, 2, 3)
    grads, laps = jax.jit(cp.get_pair_grid_grad_lap)(coords, p)
    assert grads.shape == (3, 3, 3) and laps.shape == (3, 3)
    assert np.isfinite(grads).all() and np.isfinite(laps).all()
    for i in range(3):
        for j in range(3):
            if j < 2:
                np.testing.assert_allclose(batch[i, j], cp.grad_r(coords[i], coords[j], p), atol=1e-9)
            if i != j:
                g, l = cp.get_log_grads_r1(coords[i], coords[j], p)
                np.testing.assert_allclose(grads[i, j], g, atol=1e-9)
                np.testing.assert_allclose(laps[i, j], l, atol=1e-8)


def _check_parameter_gradients_through_spatial_derivatives(mol):
    cp = CP.create(mol)
    p = varied_params(cp)
    ref = lambda params: reference(cp, R1, R2, params)
    for actual, expected in zip(jax.tree.leaves(cp.grad_params(R1, R2, p)), jax.tree.leaves(jax.grad(ref)(p))):
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    def objective(params):
        g, l = cp.get_log_grads_r1(R1, R2, params)
        return jnp.sum(g*g) + l
    def oracle(params):
        f = lambda r: reference(cp, r, R2, params)
        g = jax.grad(f)(R1)
        return jnp.sum(g*g) + jnp.trace(jax.hessian(f)(R1))
    got, want = jax.jit(jax.grad(objective))(p), jax.grad(oracle)(p)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-9)


def _check_bh_composition_and_vmc_integration(mol, bh_class):
    cp = CP.create(mol, include_cusp=False)
    bh = bh_class.create(mol)
    composite = CompositeJastrow.create([bh, cp])
    params = [bh.init_params(), varied_params(cp)]
    np.testing.assert_allclose(composite._compute(R1, R2, params),
                               bh._compute(R1, R2, params[0]) + cp._compute(R1, R2, params[1]))
    coords = jnp.stack([R1, R2, R1 + 0.7])
    sj = SimpleNamespace(jastrow=composite)
    actual_g, actual_l = compute_jastrow_terms(sj, coords, params)
    def total_u(x):
        return sum(composite._compute(x[i], x[j], params)
                   for i in range(3) for j in range(i))
    expected_g = jax.grad(total_u)(coords)
    hessian = jax.hessian(total_u)(coords)
    expected_l = jnp.array([jnp.trace(hessian[i, :, i, :]) for i in range(3)])
    expected_l += jnp.sum(expected_g**2, axis=1)
    np.testing.assert_allclose(actual_g, expected_g, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(actual_l, expected_l, rtol=1e-8, atol=1e-9)


def _check_smoothed_coalescence_and_cusp_limit(mol):
    cp = CP.create(mol, epsilon=1e-8)
    p = varied_params(cp)
    for r1, r2 in [(R1, R1), (cp.nuclear_pos[0], R2)]:
        g, lap = cp.get_log_grads_r1(r1, r2, p)
        assert np.isfinite(g).all() and np.isfinite(lap)
        expected = jnp.trace(jax.hessian(lambda x: reference(cp, x, r2, p))(r1))
        np.testing.assert_allclose(lap, expected, rtol=1e-7, atol=1e-7)
    cp = cp.replace(epsilon=0.0)
    p = {**p, "cp_C": jnp.zeros_like(p["cp_C"]), "cp_bias": jnp.zeros_like(p["cp_bias"])}
    # One-sided cusp away from the undefined exact zero-norm derivative.
    slope = cp.grad_r(R1 + jnp.array([1e-6, 0, 0]), R1, p)[0]
    np.testing.assert_allclose(slope, 0.5, atol=2e-6)


class TestCP(unittest.TestCase):
    def test_finite_differences_and_nucleus_counts(self):
        for count in (1, 3):
            with self.subTest(nuclei=count):
                positions = np.array([[0.0, 0.0, 1.4*i] for i in range(count)])
                geometry = SimpleNamespace(atom_coords=lambda: positions)
                cp = CP.create(geometry, use_cubic_features=True)
                params = varied_params(cp)
                self.assertEqual(params["cp_U"].shape[0], count)
                grad, lap = cp.get_log_grads_r1(R1, R2, params)
                h = 1e-4
                steps = jnp.eye(3) * h
                center = cp._compute(R1, R2, params)
                plus = jnp.array([cp._compute(R1+d, R2, params) for d in steps])
                minus = jnp.array([cp._compute(R1-d, R2, params) for d in steps])
                np.testing.assert_allclose(grad, (plus-minus)/(2*h), rtol=2e-6, atol=1e-8)
                np.testing.assert_allclose(lap, jnp.sum(plus-2*center+minus)/h**2,
                                           rtol=2e-5, atol=2e-7)

    @classmethod
    def setUpClass(cls):
        cls.molecule = mol()

    def test_features_and_derivatives(self):
        for degree, cubic, manual, raw in itertools.product([1, 3], [False, True], [False, True], [False, True]):
            with self.subTest(degree=degree, cubic=cubic, manual=manual, raw=raw):
                _check_values_symmetry_and_derivatives(self.molecule, cubic, manual, raw, degree)

    def test_scales(self):
        for learnable, log in [(True, False), (True, True), (False, False)]:
            with self.subTest(learnable=learnable, log=log):
                _check_initialization_and_scale(self.molecule, learnable, log)

    def test_jit(self):
        _check_dataclass_and_jit(self.molecule)

    def test_batches(self):
        _check_batch_and_pair_grid(self.molecule)

    def test_parameter_derivatives(self):
        _check_parameter_gradients_through_spatial_derivatives(self.molecule)

    def test_composition(self):
        for bh_class in (BoysHandy, BoysHandyAnalytical):
            with self.subTest(bh_class=bh_class.__name__):
                _check_bh_composition_and_vmc_integration(self.molecule, bh_class)

    def test_cusp(self):
        _check_smoothed_coalescence_and_cusp_limit(self.molecule)

    def test_invalid_configuration(self):
        for options in [
            {"cp_degree": 0}, {"cp_rank": -1}, {"epsilon": -1},
            {"epsilon": float("nan")}, {"cp_scale_init": float("inf")},
            {"use_log_scale": True, "cp_scale_init": 0}, {"use_raw_distances": True},
        ]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                CP.create(self.molecule, **options)


if __name__ == "__main__":
    unittest.main()
