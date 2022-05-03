import numpy as np

from utils.utils import (
  close_stream,
  get_egocentric_traj,
  store_pred_stream,
  stream_traj,
  compute_ADE,
  compute_FDE
)

import erdos
from erdos.operator import OperatorConfig
from erdos.streams import ExtractStream, IngestStream

from pylot.prediction.flags import flags
from pylot.prediction.linear_predictor_operator import LinearPredictorOperator

from verifai.monitor import multi_objective_monitor


class ADE_FDE(multi_objective_monitor):
    """Specification monitor that uses the Average Displacement Error
       and Final Displacement Error metrics.

    Args:
        pred_radius (py:class:int): Radius of vehicles to target.
        timepoint (py:class:int): Timestep at which to start prediction.
        past_steps (py:class:int): Number of timesteps to supply model.
        future_steps (py:class:int): Number of timesteps to receive from model.
    """

    def __init__(self, pred_radius=100, timepoint=20, past_steps=20, future_steps=15):
        assert timepoint >= past_steps, 'Timepoint must be at least the number of past steps!'
        assert past_steps >= future_steps, 'Must track at least as many steps as we predict!'
        
        flags.FLAGS.__delattr__('prediction_radius')
        flags.FLAGS.__delattr__('prediction_num_past_steps')
        flags.FLAGS.__delattr__('prediction_num_future_steps')
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
            traj_stream = IngestStream()
            [pred_stream] = erdos.connect(
                LinearPredictorOperator, OperatorConfig(name='linear_predictor_operator'),
                [traj_stream], flags.FLAGS
            )
            extract_stream = ExtractStream(pred_stream)
            driver_handle = erdos.run_async()
            try:
              stream_traj(traj_stream, timepoint, past_steps, traj)
              close_stream(traj_stream, timepoint, is_top=True)
              preds = store_pred_stream(extract_stream)

              # Dictionary mapping agent IDs to ADEs/FDEs
              ADEs, FDEs = {}, {}
              for agent_id, pred in preds.items():
                  gt = gts[agent_id]
                  ADEs[agent_id] = compute_ADE(pred, gt)
                  FDEs[agent_id] = compute_FDE(pred, gt)


              print(f'ADEs: {ADEs}, FDEs: {FDEs}')
              ADE_vals, FDE_vals = ADEs.values(), FDEs.values()
              minADE, minFDE = min(ADE_vals), min(FDE_vals)
              print(f'minADE: {minADE}, minFDE: {minFDE}')
              AADE, AFDE = sum(ADE_vals)/len(ADE_vals), sum(FDE_vals)/len(FDE_vals)
              print(f'AADE: {AADE}, AFDE: {AFDE}')

              rho = (minADE, minFDE, AADE, AFDE)
              driver_handle.shutdown()
              return rho
              
            except:
                driver_handle.shutdown()

        super().__init__(specification, priority_graph=None, linearize=False)
