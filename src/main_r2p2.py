import absl.app
import json
import os
import subprocess
import time
from dotmap import DotMap

from metrics.metrics_r2p2 import ADE_FDE
from utils.utils import announce

import scenic.core.errors as errors; errors.showInternalBacktrace = True

from verifai.samplers.scenic_sampler import ScenicSampler
from verifai.scenic_server import ScenicServer
from verifai.falsifier import generic_falsifier, generic_parallel_falsifier


PYLOT_PORT = 8000

def main(argv):
    # Load user configurations
    config_path = 'configs/config_r2p2.json'
    with open(config_path, 'r') as file:
        config = json.load(file)

    # Assign specification parameters
    lidar_filepath = config['sensor_output_dir'] + '/lidar.json'
    sensor_config_filepath = config['sensor_config_path']
    threshADE = float(config['ADE_threshold'])
    threshFDE = float(config['FDE_threshold'])
    pred_radius = 500  # this shouldn't change
    past_steps = int(config['past_steps'])
    future_steps = int(config['future_steps'])
    timepoint = int(config['timepoint'])
    # Assign platform parameters
    sampler_type = config['sampler_type']
    num_workers = int(config['parallel_workers'])
    num_iters = int(config['simulations_per_scenario'])
    output_dir = config['output_dir']
    max_steps = int(config['time_per_simulation'])
    inputs = config['input']
    headless = bool(config['headless_mode'])

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
    falsifier_cls = generic_parallel_falsifier if is_parallel else generic_falsifier
    monitor = ADE_FDE(lidar_filepath=lidar_filepath, sensor_config_filepath=sensor_config_filepath,
                      threshADE=threshADE, threshFDE=threshFDE, pred_radius=pred_radius,
                      timepoint=timepoint, past_steps=past_steps, future_steps=future_steps,
                      pylot_port=PYLOT_PORT)

    # Iterate over all Scenic programs
    for scenic_path in config['scenic_programs']:
        announce(f'RUNNING SCENIC PROGRAM {scenic_path}')
        sampler = ScenicSampler.fromScenario(scenic_path, **params)
        falsifier = falsifier_cls(monitor=monitor, sampler_type=sampler_type,
                                  sampler=sampler, falsifier_params=falsifier_params,
                                  server_options=server_options, server_class=ScenicServer)
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

if __name__ == '__main__':
    print('Killing all processes at port', PYLOT_PORT)
    subprocess.run(['fuser', '-k', f'{PYLOT_PORT}/tcp'])
    absl.app.run(main)
    subprocess.run(['fuser', '-k', f'{PYLOT_PORT}/tcp'])