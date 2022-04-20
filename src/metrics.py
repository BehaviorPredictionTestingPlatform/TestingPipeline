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
                 num_preds=1, parallel=False, debug=False):

        assert timepoint >= past_steps, 'Timepoint must be at least the number of past steps!'
        assert past_steps >= future_steps, 'Must track at least as many steps as we predict!'

        flags.DEFINE_integer('prediction_num_past_steps', past_steps, '')
        flags.DEFINE_integer('prediction_num_future_steps', future_steps, '')

        self.num_objectives = 2

        def specification(simulation):
            worker_num = simulation.worker_num if parallel else 0
            traj = simulation.trajectory
            num_agents = len(traj[0])
            past_traj = traj[timepoint-past_steps:timepoint]
            gt_traj = traj[timepoint:timepoint+future_steps]

            gts = np.asarray([(tj[-1][0], tj[-1][1]) for tj in gt_traj])
            gt_len = len(gts)

            # Dictionary mapping agent_ids to ADE/FDE list (to support multi-modal predictions)
            ADEs, FDEs = {}, {}

            if debug:
                print(f'ADE Threshold: {threshADE}, FDE Threshold: {threshFDE}')
                plt.plot([gt[-1][0] for gt in traj], [gt[-1][1] for gt in traj], color='black')
                plt.plot([gt[-1][0] for gt in past_traj], [gt[-1][1] for gt in past_traj], color='blue')
                plt.plot([gt[-1][0] for gt in gt_traj], [gt[-1][1] for gt in gt_traj], color='yellow')

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

            # Compute metrics from predicted and ground truth trajectories
            for pred_num in range(num_preds):

                # Extract predicted trajectories from CSV file
                output_csv_path = f'{out_dir}/pred_{worker_num}_{pred_num}.csv'
                preds = np.genfromtxt(output_csv_path, delimiter=',', skip_header=1)
                pred_len = preds.shape[0]
                if gt_len < pred_len:
                    preds = preds[:gt_len]

                # Compute ADE and FDE metrics
                ADE = float(
                    sum(
                        math.sqrt(
                            (preds[i, 0] - gts[i, 0]) ** 2
                            + (preds[i, 1] - gts[i, 1]) ** 2
                        )
                        for i in range(min(pred_len, gt_len))
                    ) / pred_len
                )
                FDE = math.sqrt(
                    (preds[-1, 0] - gts[-1, 0]) ** 2
                    + (preds[-1, 1] - gts[-1, 1]) ** 2
                )
                ADEs.append(ADE)
                FDEs.append(FDE)

                if debug:
                    print(f'ADE: {ADE}, FDE: {FDE}')
                    p = pd.read_csv(f'{model_path}/results/lanegcn/predictions_{worker_num}_{pred_num}.csv')
                    plt.plot(p['X'], p['Y'], color='green')

            minADE, minFDE = min(ADEs), min(FDEs)
            print(f'minADE: {minADE}, minFDE: {minFDE}')
            rho = (threshADE - minADE, threshFDE - minFDE)

            if debug:
                plt.show()

            return rho

        super().__init__(specification, priority_graph=None, linearize=False)
