import math
import numpy as np
import json

from erdos import Timestamp, WatermarkMessage

from pylot.drivers.sensor_setup import LidarSetup
from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.messages import ObstacleTrajectoriesMessage, PointCloudMessage
from pylot.perception.point_cloud import PointCloud
from pylot.perception.tracking.obstacle_trajectory import ObstacleTrajectory
from pylot.utils import Location, Rotation, Transform


def announce(message):
    m = f'* {message} *'
    border = '*' * len(m)
    print(border)
    print(m)
    print(border)

def get_egocentric_traj(traj):
    """Redefines all points to the ego vehicle's frame of reference."""
    egocentric_traj = []
    for timestep in traj:
        new_timestep = []
        ego_x, ego_y, _ = timestep[0]
        for (x, y, yaw) in timestep[1:]:
          new_timestep.append((x-ego_x, y-ego_y, yaw))
        egocentric_traj.append(new_timestep)
    return egocentric_traj

def close_stream(stream, timepoint, is_top=False):
    stream.send(WatermarkMessage(Timestamp(coordinates=[timepoint], is_top=is_top)))

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

    # Stream agent trajectories
    for _, data in obs_trajs.items():
        timestamp = Timestamp(coordinates=data['coordinates'])
        obs_traj = [ObstacleTrajectory(data['obstacle'], data['trajectory'])]
        msg = ObstacleTrajectoriesMessage(timestamp, obs_traj)
        stream.send(msg)

def get_lidar_setup(sensor_config_filepath):
    # Supported `LidarSetup` settings
    LIDAR_SETTINGS = {'CHANNELS': 'channels',
                      'LOWER_FOV': 'lower_fov',
                      'PPS': 'points_per_second',
                      'RANGE': 'range',
                      'ROTATION_FREQUENCY': 'rotation_frequency',
                      'UPPER_FOV': 'upper_fov'}

    # Extract LiDAR configurations
    sensor_config_file = open(sensor_config_filepath, 'r')
    sensors = json.load(sensor_config_file)
    sensor_configs = {'transform': None}
    for sensor in sensors:
      if sensor['type'] != 'lidar':
        continue
      x, y, z = sensor['transform']
      sensor_configs['transform'] = Transform(location=Location(x=x, y=y, z=z),
                                              rotation=Rotation())
      for setting, val in sensor['settings'].items():
        if setting not in LIDAR_SETTINGS:
          continue
        sensor_configs[LIDAR_SETTINGS[setting]] = val
    sensor_config_file.close()
    assert sensor_configs['transform'] is not None, 'Missing LiDAR transform in sensor config file!'
    lidar_setup = LidarSetup(name='lidar',
                             lidar_type='sensor.lidar.ray_cast',
                             legacy=False,  # Using CARLA v.0.9.10.1-74-g8f1b401e
                             **sensor_configs)
    return lidar_setup

def stream_pc(stream, timepoint, lidar_filepath, lidar_setup):
    # Extract LiDAR points as numpy arrays
    lidar_file = open(lidar_filepath, 'r')
    lidar_data = json.load(lidar_file)
    lidar_file.close()
    points = np.array([[d[0],d[1],d[2]] for d in lidar_data[0]])

    # Stream point cloud at timestep one before timepoint
    timestamp = Timestamp(coordinates=[timepoint-1])
    pc = PointCloud(points, lidar_setup)
    msg = PointCloudMessage(timestamp, pc)
    stream.send(msg)

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
