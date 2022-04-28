import math
import numpy as np

from erdos import Timestamp, WatermarkMessage

from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.messages import ObstacleTrajectoriesMessage
from pylot.perception.tracking.obstacle_trajectory import ObstacleTrajectory
from pylot.utils import Location, Rotation, Transform


def announce(message):
    m = f'* {message} *'
    border = '*' * len(m)
    print(border)
    print(m)
    print(border)

def stream_traj(stream, timepoint, past_steps, traj):
    # Dictionary mapping agents to trajectory data
    obs_trajs = {}
    for timestamp in range(timepoint-past_steps, timepoint):
        for agent_id, transform in enumerate(traj[timestamp]):
            # Add new agent to dictionary and initalize its fields
            if agent_id not in obs_trajs:
                obs_trajs[agent_id] = {}
                obs_trajs[agent_id]['obstacle'] = Obstacle(None, 'vehicle',
                                                           0.0, id=agent_id)
                obs_trajs[agent_id]['coordinates'] = []
                obs_trajs[agent_id]['trajectory'] = []
            # Record timestamp and trajectory data
            x, y, yaw = transform
            obs_trajs[agent_id]['coordinates'].append(timestamp)
            obs_trajs[agent_id]['trajectory'].append(
                Transform(location=Location(x=x, y=y),
                          rotation=Rotation(yaw=yaw)))

    # Stream each agent's trajectory
    for _, data in obs_trajs.items():
        timestamp = Timestamp(coordinates=data['coordinates'])
        obs_traj = [ObstacleTrajectory(data['obstacle'], data['trajectory'])]
        msg = ObstacleTrajectoriesMessage(timestamp, obs_traj)
        stream.send(msg)

    # Send watermark message to indicate completion
    top_timestamp = Timestamp(coordinates=[timepoint], is_top=True)
    watermark_msg = WatermarkMessage(top_timestamp)
    stream.send(watermark_msg)

def store_pred_stream(stream):
    # Dictionary mapping agent IDs to predicted trajectories
    preds = {}
    while not isinstance(msg := stream.read(), WatermarkMessage):
        for pred in msg.predictions:
            agent_id = pred.obstacle_trajectory.obstacle.id
            if agent_id not in preds:
                preds[agent_id] = []
            for _, traj in enumerate(pred.predicted_trajectory):
                x, y = traj.location.x, traj.location.y
                yaw = traj.rotation.yaw
                preds[agent_id].append((x, y, yaw))
    for agent_id, pred in preds.items():
        preds[agent_id] = np.array(pred)
    return preds

def compute_ADE(pred, gt):
    pred_len, gt_len = len(pred), len(gt)
    return float(
        sum(
            math.sqrt(
                (pred[i, 0] - gt[i, 0]) ** 2
                + (pred[i, 1] - gt[i, 1]) ** 2
            )
            for i in range(min(pred_len, gt_len))
        ) / pred_len
    )

def compute_FDE(pred, gt):
    return math.sqrt(
        (pred[-1, 0] - gt[-1, 0]) ** 2
        + (pred[-1, 1] - gt[-1, 1]) ** 2
    )
