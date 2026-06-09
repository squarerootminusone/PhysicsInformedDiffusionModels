"""Run ONE training in an isolated process, so reduce-overhead cudagraphs are safe
(the RNG/graph-capture state never has to survive a second compiled model).

Usage:  python run_one_trial.py '<overrides-json>' <result_file>
Writes the objective (best residual_mean_abs_samples) to <result_file>.
"""
import sys, json
import main

overrides = json.loads(sys.argv[1])
result_path = sys.argv[2]
overrides.setdefault('compile_mode', 'reduce-overhead')   # one model per process -> cudagraphs OK

value = main.train(overrides, trial=None)

with open(result_path, 'w') as f:
    f.write(repr(float(value)))
print(f'OBJECTIVE={value}', flush=True)
