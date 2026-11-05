"""Compare generic and shared CPA batching, including compiler memory estimates."""
import json
import statistics
import time
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from pytc.jastrow import CPAnalytical

jax.config.update('jax_enable_x64', True)

def compare(fn, models, args, label):
    compiled = []
    rows = []
    for model in models:
        start = time.perf_counter()
        program = jax.jit(fn).lower(model, *args).compile()
        elapsed = time.perf_counter()-start
        memory = program.memory_analysis()
        compiled.append(program)
        rows.append(dict(shared=model.use_shared_pair_grid, compile_seconds=elapsed,
                         temp_bytes=memory.temp_size_in_bytes,
                         argument_bytes=memory.argument_size_in_bytes,
                         output_bytes=memory.output_size_in_bytes))
    outputs = [jax.block_until_ready(program(model, *args)) for program, model in zip(compiled, models)]
    for a, b in zip(jax.tree.leaves(outputs[0]), jax.tree.leaves(outputs[1])):
        np.testing.assert_allclose(a, b, rtol=2e-8, atol=2e-9)
    for _ in range(5):
        for program, model in zip(compiled, models):
            jax.block_until_ready(program(model, *args))
    samples = [[], []]
    for iteration in range(21):
        # Alternate timing order to reduce warm-up/order bias.
        for i in ((0, 1) if iteration % 2 else (1, 0)):
            start = time.perf_counter()
            jax.block_until_ready(compiled[i](models[i], *args))
            samples[i].append(time.perf_counter()-start)
    for row, timings in zip(rows, samples):
        row['median_ms'] = 1000*statistics.median(timings)
    print(json.dumps(dict(workload=label, results=rows,
                         speedup=rows[0]['median_ms']/rows[1]['median_ms'])), flush=True)

def main():
    print(json.dumps(dict(device=str(jax.devices()[0]), jax=jax.__version__, degree=3, rank=4, dtype='float64')), flush=True)
    for atoms in (3, 12):
        geom = SimpleNamespace(atom_coords=lambda: np.arange(atoms*3).reshape(atoms, 3)*0.15)
        generic = CPAnalytical.create(geom, use_shared_pair_grid=False)
        models = [generic, generic.replace(use_shared_pair_grid=True)]
        p = generic.init_params(cp_init_scale=0.1)
        x = jax.random.normal(jax.random.PRNGKey(81), (256, 3)) + 0.3
        for n in (4, 16, 64, 128):
            compare(lambda model, x, p: model.get_pair_grid_grad_lap(x, p),
                    models, (x[:n], p), f'pair_grad_lap_N{n}_M{atoms}')
        for a, b in ((8, 16), (64, 128), (128, 256)):
            compare(lambda model, x, y, p: model.grad_r_batch(x, y, p),
                    models, (x[:a], x[:b]+0.1, p), f'rectangular_grad_{a}x{b}_M{atoms}')
        for n in (16, 64):
            def objective(model, params, coords):
                g, lap = model.get_pair_grid_grad_lap(coords, params)
                mask = 1-jnp.eye(len(coords))
                g = jnp.sum(g*mask[..., None], axis=1)
                return jnp.sum(lap*mask) + jnp.sum(g*g)
            compare(jax.grad(objective, argnums=1), models, (p, x[:n]),
                    f'parameter_gradient_N{n}_M{atoms}')


if __name__ == '__main__':
    main()
