import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

import erdos
from erdos.operator import OperatorConfig
from erdos.streams import ExtractStream, IngestStream

from pylot.prediction.flags import flags
from pylot.prediction.linear_predictor_operator import LinearPredictorOperator

from verifai.monitor import multi_objective_monitor


class Pylot_ADE_FDE(multi_objective_monitor):
    """Specification monitor that uses the Average Displacement Error
       and Final Displacement Error metrics.

    Args:
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

    def __init__(self, threshADE=0.5, threshFDE=1.0,
                 timepoint=20, past_steps=20, future_steps=15,
                 parallel=False, debug=False):

        assert timepoint >= past_steps, 'Timepoint must be at least the number of past steps!'
        assert past_steps >= future_steps, 'Must track at least as many steps as we predict!'

        flags.FLAGS.__delattr__('prediction_num_past_steps')
        flags.FLAGS.__delattr__('prediction_num_future_steps')
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
                        gts[agent_id] = []
                    gts[agent_id].append(transform)
            for agent_id, gt in gts.items():
                gts[agent_id] = np.array(gt)

            if debug:
                print(f'ADE Threshold: {threshADE}, FDE Threshold: {threshFDE}')
                plt.plot([gt[-1][0] for gt in traj], [gt[-1][1] for gt in traj], color='black')
                plt.plot([gt[-1][0] for gt in past_trajs], [gt[-1][1] for gt in past_trajs], color='blue')
                plt.plot([gt[-1][0] for gt in gt_trajs], [gt[-1][1] for gt in gt_trajs], color='yellow')

            # Run behavior prediction model
            traj_stream = IngestStream()
            [pred_stream] = erdos.connect(
                LinearPredictorOperator, OperatorConfig(name='linear_predictor_operator'),
                [traj_stream], flags.FLAGS
            )
            extract_stream = ExtractStream(pred_stream)
            driver_handle = erdos.run_async()
            stream_traj(traj_stream, timepoint, past_steps, traj)
            preds = store_pred_stream(extract_stream)
            # NOTE: I would like to call driver_handle.wait() below instead of
            #       sleep+shutdown, but the process hangs for some reason
            time.sleep(0.1)
            driver_handle.shutdown()

            # Dictionary mapping agent IDs to ADEs/FDEs
            ADEs, FDEs = {}, {}
            for agent_id, pred in preds.items():
                gt = gts[agent_id]
                ADEs[agent_id] = compute_ADE(pred, gt)
                FDEs[agent_id] = compute_FDE(pred, gt)

            if debug:
                print(f'ADE: {ADE}, FDE: {FDE}')
                p = pd.read_csv(output_csv_path)
                plt.plot(p['X'], p['Y'], color='green')

            minADE, minFDE = min(ADEs.values()), min(FDEs.values())
            rho = (threshADE - minADE, threshFDE - minFDE)
            
            print(f'ADEs: {ADEs}, FDEs: {FDEs}')
            print(f'minADE: {minADE}, minFDE: {minFDE}')

            if debug:
                plt.show()

            return rho

        super().__init__(specification, priority_graph=None, linearize=False)
