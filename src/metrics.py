import absl.flags as flags
import csv
import erdos
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pylot.prediction.flags
from pylot.prediction.linear_predictor_operator import LinearPredictorOperator

from verifai.monitor import multi_objective_monitor


class ADE_FDE(multi_objective_monitor):
    """Specification monitor that uses the Average Displacement Error
       and Final Displacement Error metrics.

    Args:
        in_dir (py:class:str): Absolute dir path to write past trajectories.
        out_dir (py:class:str): Absolute dir path to write predictions.
        threshADE (py:class:float): Failure threshold for ADE metric.
        threshFDE (py:class:float): Failure threshold for FDE metric.
        timepoint (py:class:int): Timestep at which to start prediction.
        past_steps (py:class:int): Number of timesteps to supply model.
        future_steps (py:class:int): Number of timesteps to receive from model.
        num_preds (py:class:int): Number of predictions to receive from model.
        parallel (py:class:bool): Indicates if using parallelized VerifAI.
        debug (py:class:bool): Indicates if debugging mode is on.
    """

    def __init__(self, in_dir, out_dir, threshADE=0.5, threshFDE=1.0,
                 timepoint=20, past_steps=20, future_steps=15,
                 parallel=False, debug=False):

        assert timepoint >= past_steps, 'Timepoint must be at least the number of past steps!'
        assert past_steps >= future_steps, 'Must track at least as many steps as we predict!'

        flags.DEFINE_integer('prediction_num_past_steps', past_steps, '')
        flags.DEFINE_integer('prediction_num_future_steps', future_steps, '')

        self.num_objectives = 2

        def specification(simulation):
            worker_num = simulation.worker_num if parallel else 0
            traj = simulation.trajectory
            past_trajs = traj[timepoint-past_steps:timepoint]
            gt_trajs = traj[timepoint:timepoint+future_steps]
            gt_len = len(gt_trajs)

            # Dictionary mapping agent IDs to ground truth trajectories
            gts = {}
            for gt_traj in gt_trajs:
                for agent_id, transform in enumerate(gt_traj):
                    if agent_id not in gts:
                        gts[agent_id] = np.array()
                    gts[agent_id].append(transform)

            if debug:
                print(f'ADE Threshold: {threshADE}, FDE Threshold: {threshFDE}')
                plt.plot([gt[-1][0] for gt in traj], [gt[-1][1] for gt in traj], color='black')
                plt.plot([gt[-1][0] for gt in past_trajs], [gt[-1][1] for gt in past_trajs], color='blue')
                plt.plot([gt[-1][0] for gt in gt_trajs], [gt[-1][1] for gt in gt_trajs], color='yellow')

            # Write past trajectories to CSV file
            input_csv_path = f'{in_dir}/past_{worker_num}_0.csv'
            csv_file = open(input_csv_path, 'w', newline='')
            writer = csv.writer(csv_file)
            writer.writerow(['timestamp', 'agent_id', 'x', 'y', 'yaw'])
            for timestamp in range(timepoint-past_steps, timepoint):
                for agent_id, transform in enumerate(traj[timestamp]):
                    writer.writerow([timestamp, agent_id, transform[0], transform[1], transform[2]])
            csv_file.close()

            # Run behavior prediction model
            [tracking_stream] = erdos.connect(FromCsvOperator, erdos.OperatorConfig(), [], input_csv_path)
            [prediction_stream] = erdos.connect(LinearPredictorOperator, erdos.OperatorConfig(), [tracking_stream], flags.FLAGS)
            erdos.connect(ToCsvOperator, erdos.OperatorConfig(), [prediction_stream], out_dir, worker_num)
            erdos.run()

            # Extract predicted trajectories from CSV file
            output_csv_path = f'{out_dir}/pred_{worker_num}.csv'
            pred_trajs = np.genfromtxt(output_csv_path, delimiter=',', skip_header=1)
            pred_len = pred_trajs.shape[0]
            if gt_len < pred_len:
                pred_trajs = pred_trajs[:gt_len]

            # Sort by timestamp
            pred_trajs = pred_trajs[pred_trajs[:, 0].argsort()]

            # Dictionary mapping agent IDs to predicted trajectories
            preds = {}
            for pred_traj in pred_trajs:
                _, agent_id, x, y, yaw = pred_traj
                if agent_id not in preds:
                    preds[agent_id] = np.array()
                preds[agent_id].append((x, y, yaw))

            # Dictionary mapping agent IDs to ADEs/FDEs
            ADEs, FDEs = {}, {}
            for agent_id, pred in preds.items():
                gt = gts[agent_id]
                ADEs[agent_id] = float(
                    sum(
                        math.sqrt(
                            (pred[i, 0] - gt[i, 0]) ** 2
                            + (pred[i, 1] - gt[i, 1]) ** 2
                        )
                        for i in range(min(pred_len, gt_len))
                    ) / pred_len
                )
                FDEs[agent_id] = math.sqrt(
                    (pred[-1, 0] - gt[-1, 0]) ** 2
                    + (pred[-1, 1] - gt[-1, 1]) ** 2
                )

            if debug:
                print(f'ADE: {ADE}, FDE: {FDE}')
                p = pd.read_csv(f'{model_path}/results/lanegcn/predictions_{worker_num}.csv')
                plt.plot(p['X'], p['Y'], color='green')

            minADE, minFDE = min(ADEs.values()), min(FDEs.values())
            print(f'minADE: {minADE}, minFDE: {minFDE}')
            rho = (threshADE - minADE, threshFDE - minFDE)

            if debug:
                plt.show()

            return rho

        super().__init__(specification, priority_graph=None, linearize=False)
