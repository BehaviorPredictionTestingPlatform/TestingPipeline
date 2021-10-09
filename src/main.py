import json
import math
import os
import time
from dotmap import DotMap

from metrics import ADE_FDE
from utils import announce

from verifai.samplers.scenic_sampler import ScenicSampler
from verifai.scenic_server import ScenicServer
from verifai.falsifier import generic_falsifier, generic_parallel_falsifier
from verifai.monitor import multi_objective_monitor

# Load user configurations
config_path = './config.json'
with open(config_path, 'r') as file:
    config = json.load(file)

# Assign platform parameters
model_path = config['behavior_prediction_model']
timepoint = config['timepoint']
sampler_type = config['sampler_type']
num_workers = config['parallel_workers']
num_iters = config['simulations_per_scenario']
output_dir = config['output_dir']
max_steps = config['time_per_simulation']
inputs = config['input']
headless = config['headless_mode']

# Store Scenic and VerifAI parameters
is_parallel = num_workers > 1
params = {'verifaiSamplerType': sampler_type} if sampler_type else {}
params['render'] = not headless
params['record'] = any([i in inputs for i in \
    ('rgb', 'depth', 'semantic_segmentation', 'lidar', 'radar')])
falsifier_params = DotMap(
    n_iters=num_iters,
    save_error_table=True,
    save_safe_table=True,
    max_time=None,
)
server_options = DotMap(maxSteps=max_steps, verbosity=0)
monitor = ADE_FDE(model_path, timepoint=timepoint, parallel=is_parallel)
falsifier_cls = generic_parallel_falsifier if is_parallel else generic_falsifier

# Iterate over all Scenic programs
for scenic_path in config['scenic_programs']:
    announce(f'RUNNING SCENIC PROGRAM {scenic_path}')
    sampler = ScenicSampler.fromScenario(scenic_path, **params)
    falsifier = falsifier_cls(sampler=sampler, falsifier_params=falsifier_params,
                                server_class=ScenicServer, server_options=server_options,
                                monitor=monitor, scenic_path=scenic_path,
                                scenario_params=params, num_workers=num_workers)
    t0 = time.time()
    falsifier.run_falsifier()
    t = time.time() - t0
    print(f'\nGenerated {len(falsifier.samples)} samples in {t} seconds with {falsifier.num_workers} workers')
    print(f'Number of counterexamples: {len(falsifier.error_table.table)}')
    print(f'Confidence interval: {falsifier.get_confidence_interval()}')
    
    tables = []
    if falsifier_params.save_error_table:
        tables.append(falsifier.error_table.table)
        tables.append(falsifier.safe_table.table)
    root, _ = os.path.splitext(scenic_path)
    for i, df in enumerate(tables):
        outfile = root.split('/')[-1]
        if is_parallel:
            outfile += '_parallel'
        if sampler_type:
            outfile += f'_{sampler_type}'
        if i == 0:
            outfile += '_error'
        else:
            outfile += '_safe'
        outfile += '.csv'
        outpath = os.path.join(output_dir, outfile)
        announce(f'SAVING OUTPUT TO {outpath}')
        df.to_csv(outpath)
