import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import subprocess

from verifai.monitor import multi_objective_monitor

class ADE_FDE(multi_objective_monitor):
    def __init__(self, model_path, thresholds=(0.1, 1), timepoint=20, parallel=False, debug=False):
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
