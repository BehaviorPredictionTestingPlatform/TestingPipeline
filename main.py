import argparse
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import subprocess
import time
from dotmap import DotMap

from verifai.samplers.scenic_sampler import ScenicSampler
from verifai.scenic_server import ScenicServer
from verifai.falsifier import generic_falsifier, generic_parallel_falsifier
from verifai.monitor import multi_objective_monitor

class ADE_FDE(multi_objective_monitor):
    def __init__(self, model_path, thresholds=(2, 2), timepoint=20, parallel=False, debug=False):
        priority_graph = None
        self.model_path = model_path
        self.num_objectives = 2
        self.parallel = parallel
        self.debug = debug
        self.thresholds = thresholds
        self.timepoint = timepoint
        assert len(thresholds) == self.num_objectives, f'Must include {self.num_objectives} threshold values!'
        assert timepoint >= 20, 'Must allow at least 20 timesteps of past trajectories!'

        def specification(simulation):
            worker_num = simulation.worker_num if self.parallel else 0
            traj = simulation.trajectory
            num_agents = len(traj[0])
            hist_traj = traj[timepoint-20:timepoint]
            gt_traj = traj[timepoint:timepoint+15]
            gts = np.asarray([(tj[-1][0], tj[-1][1]) for tj in gt_traj])
            gt_len = len(gts)
            threshADE, threshFDE = self.thresholds

            if self.debug:
                print(f'ADE Threshold: {threshADE}, FDE Threshold: {threshFDE}')
                plt.plot([gt[-1][0] for gt in traj], [gt[-1][1] for gt in traj], color='black')
                plt.plot([gt[-1][0] for gt in hist_traj], [gt[-1][1] for gt in hist_traj], color='blue')
                plt.plot([gt[-1][0] for gt in gt_traj], [gt[-1][1] for gt in gt_traj], color='yellow')

            # Process historical trajectory CSV file
            with open(f'{model_path}/dataset/test_obs/data_{worker_num}/0.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['TIMESTAMP', 'TRACK_ID', 'OBJECT_TYPE', 'X', 'Y', 'CITY_NAME'])
                track_id = '00000000-0000-0000-0000-000000000000'
                for timestamp in range(timepoint-20, timepoint):
                    for obj_type, agent_pos in enumerate(traj[timestamp]):
                        if obj_type == 0:
                            obj_type = 'AV'
                        elif obj_type == num_agents - 1:
                            obj_type = 'AGENT'
                        else:
                            obj_type = 'OTHER'
                        writer.writerow([timestamp, track_id, obj_type, agent_pos[0], agent_pos[1], 'N/A'])
            csvfile.close()

            # Run behavior prediction model
            currDir = os.path.abspath(os.getcwd())
            os.chdir(model_path)
            subprocess.run(['python', 'preprocess_data.py', '-n', '1', '-w', f'{worker_num}'])
            subprocess.run(['python', 'test.py', '-m', 'lanegcn', f'--weight={model_path}/36.000.ckpt', '--split=test', '--map_path=/maps/CARLA/Town05.xodr', f'--worker_num={worker_num}'])
            os.chdir(currDir)

            ADEs, FDEs = [], []
            for i in range(6):
                # Extract predicted trajectories from CSV file
                preds = np.genfromtxt(f'{model_path}/results/lanegcn/predictions_{worker_num}_{i}.csv', delimiter=',', skip_header=1)
                pred_len = preds.shape[0]
                if gt_len < pred_len:
                    preds = preds[:gt_len]

                # Compute metrics
                ADE = float(
                    sum(
                        math.sqrt(
                            (preds[i, 0] - gts[i, 0]) ** 2
                            + (preds[i, 1] - gts[i, 1]) ** 2
                        )
                        for i in range(min(pred_len, gt_len))
                    )
                    / pred_len
                )
                FDE = math.sqrt(
                    (preds[-1, 0] - gts[-1, 0]) ** 2
                    + (preds[-1, 1] - gts[-1, 1]) ** 2
                )
                ADEs.append(ADE)
                FDEs.append(FDE)

                if self.debug:
                    print(f'ADE: {ADE}, FDE: {FDE}')
                    p = pd.read_csv(f'{model_path}/results/lanegcn/predictions_{worker_num}_{i}.csv')
                    plt.plot(p['X'], p['Y'], color='green')

            minADE, minFDE = min(ADEs), min(FDEs)
            print(f'minADE: {minADE}, minFDE: {minFDE}')
            rho = (threshADE - minADE, threshFDE - minFDE)

            if self.debug:
                plt.show()
            return rho

        super().__init__(specification, priority_graph)

def announce(message):
    m = f'* {message} *'
    border = '*' * len(m)
    print(border)
    print(m)
    print(border)

def run_experiment(scenic_path, model_path, thresholds=None, timepoint=20,
                    sampler_type=None, num_workers=1, num_iters=10, 
                    headless=False, output_dir='outputs', debug=False):
    announce(f'RUNNING SCENIC PROGRAM {scenic_path}')
    
    parallel = num_workers > 1
    params = {'verifaiSamplerType': sampler_type} if sampler_type else {}
    params['render'] = not headless
    sampler = ScenicSampler.fromScenario(scenic_path, **params)
    falsifier_params = DotMap(
        n_iters=num_iters,
        save_error_table=True,
        save_safe_table=True,
        max_time=None,
    )
    server_options = DotMap(maxSteps=200, verbosity=0)
    if thresholds:
        monitor = ADE_FDE(model_path, thresholds=thresholds, timepoint=timepoint, parallel=parallel, debug=debug)
    else:
        monitor = ADE_FDE(model_path, timepoint=timepoint, parallel=parallel, debug=debug)

    falsifier_cls = generic_parallel_falsifier if parallel else generic_falsifier
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
    if falsifier_params.save_error_table:
        tables.append(falsifier.safe_table.table)
    root, _ = os.path.splitext(scenic_path)
    for i, df in enumerate(tables):
        outfile = root.split('/')[-1]
        if parallel:
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

    return falsifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', '-p', type=str, help='Path to Scenic program'
    )
    parser.add_argument(
        '--model', '-m', type=str, help='Path to behavior prediction model'
    )
    parser.add_argument(
        '--threshold', '-t', type=float, action='append'
    )
    parser.add_argument(
        '--timepoint', '-k', type=int, default=20, help='Timepoint to start behavior prediction'
    )
    parser.add_argument(
        '--samplerType', '-s', type=str, default=None, help='VerifAI sampler type'
    )
    parser.add_argument(
        '--numWorkers', '-n', type=int, default=1, help='Number of parallel workers'
    )
    parser.add_argument(
        '--numIters', '-i', type=int, default=10, help='Number of iterations to run'
    )
    parser.add_argument(
        '--headless', action='store_true'
    )
    parser.add_argument(
        '--debug', action='store_true'
    )
    args = parser.parse_args()

    falsifier = run_experiment(args.path, args.model, thresholds=tuple(args.threshold), timepoint=args.timepoint,
        sampler_type=args.samplerType, num_workers=args.numWorkers, num_iters=args.numIters,
        headless=args.headless, debug=args.debug)
