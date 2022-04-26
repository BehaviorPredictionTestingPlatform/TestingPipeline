"""
Implements an operator for interfacing [1] to a Pylot prediction operator.

[1] https://arxiv.org/abs/2110.14870

Authors:
  * Francis Indaheng (findaheng@berkeley.edu)
"""

import csv
import erdos

from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.messages import ObstacleTrajectoriesMessage
from pylot.perception.tracking.obstacle_trajectory import ObstacleTrajectory
from pylot.utils import Location, Rotation, Transform


class FromCsvOperator(erdos.Operator):
    """Operator to stream a CSV file of historical trajectories
       to a Pylot prediction operator.
    
    Args:
        tracking_stream (:py:class:`erdos.WriteStream`): Stream on which messages
            (:py:class:`~pylot.perception.messages.ObstacleTrajectoriesMessage`) are sent.
        csv_file_path (:py:class:str): Absolute path to CSV file to read from.
    """
    def __init__(self, tracking_stream: erdos.WriteStream, csv_file_path: str):
        self._logger = erdos.utils.setup_logging(self.config.name, self.config.log_file_name)
        self.tracking_stream = tracking_stream
        self.csv_file = open(csv_file_path, 'r')
        self.data = csv.reader(self.csv_file)

    @staticmethod
    def connect():
        tracking_stream = erdos.WriteStream()
        return [tracking_stream]

    def destroy(self):
        self._logger.warn(f'Destroying {self.config.name}')
        self.csv_file.close()

    def run(self):
        # Dictionary mapping agents to trajectory data
        obs_trajs = {}

        for i, traj in enumerate(self.data):
            if i == 0:
                assert traj == ['timestamp', 'agent_id', 'x', 'y', 'yaw']
                continue

            agent_id = int(traj[1])

            # Add new agent to dictionary and initalize its fields
            if agent_id not in obs_trajs:
                obs_trajs[agent_id] = {}
                obs_trajs[agent_id]['obstacle'] = Obstacle(None, 'vehicle',
                                                           0.0, id=agent_id)
                obs_trajs[agent_id]['coordinates'] = []
                obs_trajs[agent_id]['trajectory'] = []

            # Record timestamp and trajectory data
            timestamp = top_timestamp = int(traj[0])
            x, y, yaw = float(traj[2]), float(traj[3]), float(traj[4])
            obs_trajs[agent_id]['coordinates'].append(timestamp)
            obs_trajs[agent_id]['trajectory'].append(
                Transform(location=Location(x=x, y=y),
                          rotation=Rotation(yaw=yaw)))

        # Stream each agent's trajectory
        for _, data in obs_trajs.items():
            timestamp = erdos.Timestamp(coordinates=data['coordinates'])
            obs_traj = [ObstacleTrajectory(data['obstacle'], data['trajectory'])]
            msg = ObstacleTrajectoriesMessage(timestamp, obs_traj)
            self.tracking_stream.send(msg)

        # Send watermark message to indicate completion
        top_timestamp = erdos.Timestamp(coordinates=[top_timestamp], is_top=True)
        watermark_msg = erdos.WatermarkMessage(top_timestamp)
        self.tracking_stream.send(watermark_msg)
