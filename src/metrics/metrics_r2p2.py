import numpy as np

from utils.utils import (
  close_stream,
  get_egocentric_traj,
  get_lidar_setup,
  store_pred_stream,
  stream_pc,
  stream_traj,
  compute_ADE,
  compute_FDE
)

import erdos
from erdos.operator import OperatorConfig
from erdos.streams import ExtractStream, IngestStream

from pylot.prediction.flags import flags
from pylot.prediction.r2p2_predictor_operator import R2P2PredictorOperator

from verifai.monitor import multi_objective_monitor


class ADE_FDE(multi_objective_monitor):
    """Specification monitor that uses the Average Displacement Error
       and Final Displacement Error metrics.

    Args:
        lidar_filepath (py:class:str): Absolute path to JSON file of LiDAR data.
        sensor_config_filepath (py:class:str): Absolute path to sensor config file.
        threshADE (py:class:float): Failure threshold for ADE metric.
        threshFDE (py:class:float): Failure threshold for FDE metric.
        pred_radius (py:class:int): Radius of vehicles to target.
        timepoint (py:class:int): Timestep at which to start prediction.
        past_steps (py:class:int): Number of timesteps to supply model.
        future_steps (py:class:int): Number of timesteps to receive from model.
        pylot_port (py:class:int): Port number used by Pylot processes.
    """

    def __init__(self, lidar_filepath, sensor_config_filepath,
                 threshADE=0.5, threshFDE=1.0, pred_radius=100,
                 timepoint=20, past_steps=20, future_steps=15,
                 pylot_port=8000):

        assert timepoint >= past_steps, 'Timepoint must be at least the number of past steps!'
        assert past_steps >= future_steps, 'Must track at least as many steps as we predict!'
        
        flags.FLAGS.__delattr__('r2p2_model_path')
        flags.FLAGS.__delattr__('prediction_radius')
        flags.FLAGS.__delattr__('prediction_num_past_steps')
        flags.FLAGS.__delattr__('prediction_num_future_steps')
        flags.DEFINE_string('r2p2_model_path',
          '/home/carla_challenge/Desktop/kaleab/pylot/dependencies/models/prediction/r2p2/r2p2-model.pt', '')
        flags.DEFINE_integer('prediction_radius', pred_radius, '')
        flags.DEFINE_integer('prediction_num_past_steps', past_steps, '')
        flags.DEFINE_integer('prediction_num_future_steps', future_steps, '')

        self.num_objectives = 2

        def specification(simulation):
            traj = get_egocentric_traj(simulation.trajectory)
            gt_trajs = traj[timepoint:timepoint+future_steps]

            # Dictionary mapping agent IDs to ground truth trajectories
            gts = {}
            for gt_traj in gt_trajs:
                for agent_id, transform in enumerate(gt_traj):
                    if agent_id not in gts:
                        gts[agent_id] = []
                    gts[agent_id].append(transform)
            for agent_id, gt in gts.items():
                gts[agent_id] = np.array(gt)

            # Run behavior prediction model
            pc_stream, traj_stream = IngestStream(), IngestStream()
            [pred_stream] = erdos.connect(
                R2P2PredictorOperator, OperatorConfig(name='r2p2_predictor_operator'),
                [pc_stream, traj_stream], flags.FLAGS, lidar_setup := get_lidar_setup(sensor_config_filepath)
            )
            extract_stream = ExtractStream(pred_stream)
            driver_handle = erdos.run_async()
            try:
              stream_pc(pc_stream, timepoint, lidar_filepath, lidar_setup)
              stream_traj(traj_stream, timepoint, past_steps, traj)
              close_stream(pc_stream, timepoint)
              close_stream(traj_stream, timepoint)
              preds = store_pred_stream(extract_stream)

              # Dictionary mapping agent IDs to ADEs/FDEs
              ADEs, FDEs = {}, {}
              for agent_id, pred in preds.items():
                  gt = gts[agent_id]
                  ADEs[agent_id] = compute_ADE(pred, gt)
                  FDEs[agent_id] = compute_FDE(pred, gt)

              print(f'ADEs: {ADEs}, FDEs: {FDEs}')
              minADE, minFDE = min(ADEs.values()), min(FDEs.values())
              print(f'minADE: {minADE}, minFDE: {minFDE}')

              rho = (minADE, minFDE)
              driver_handle.shutdown()
              return rho
              
            except:
                driver_handle.shutdown()

        super().__init__(specification, priority_graph=None, linearize=False)
