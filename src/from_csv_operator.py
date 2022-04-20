"""
Implements an operator for interfacing [1] to the Pylot LinearPredictorOperator.

[1] https://arxiv.org/abs/2110.14870

Authors:
  * Francis Indaheng (findaheng@berkeley.edu)
"""

import csv
import erdos

from pylot.perception.detection import Obstacle
from pylot.perception.messages import ObstacleTrajectoriesMessage
from pylot.perception.tracking import ObstacleTrajectory
from pylot.utils import Location, Rotation, Transform


class FromCsvOperator(erdos.Operator):
    """Operator to stream a CSV file of historical trajectories
       to the Pylot LinearPredictorOperator.
    
    Args:
        csv_file_path (:py:class:str): Absolute path to CSV file to read from.
    """
    def __init__(self, csv_file_path: str):
        self.csv_file = open(csv_file_path, 'r'):
        self.data = csv.reader(self.csv_file)
        self.tracking_stream = erdos.WriteStream()

    @staticmethod
    def connect():
        return [self.tracking_stream]

    def destroy(self):
        self.csv_file.close()

    def run(self):
        traj_keys, traj_data = self.data[0], self.data[1:]
        assert traj_keys == ['timestamp', 'agent_id', 'x', 'y', 'yaw']
        
        # Dictionary mapping agents to trajectory data
        obs_trajs = {}

        for traj in traj_data:
            agent_id = int(traj[1])

            # Add new agent to dictionary and initalize its fields
            if agent_id not in obs_traj:
                obs_trajs[agent_id] = {}
                obs_trajs[agent_id]['obstacle'] = Obstacle(None, 'vehicle',
                                                           0.0, id=agent_id)
                obs_trajs[agent_id]['coordinates'] = []
                obs_trajs[agent_id]['trajectory'] = []

            # Record timestamp and trajectory data
            obs_trajs[agent_id]['coordinates'].append(traj[0])
            obs_trajs[agent_id]['trajectory'].append(
                Transform(location=Location(x=traj[2], y=traj[3]),
                          rotation=Rotation(yaw=traj[4])))

        # Stream each agent's trajectory
        for _, data in obs_trajs.items():
            timestamp = erdos.Timestamp(coordinates=data['coordinates'])
            obs_traj = ObstacleTrajectory(data['obstacle'], data['trajectory'])
            msg = ObstacleTrajectoriesMessage(timestamp, obs_traj)
            self.tracking_stream.send(msg)
