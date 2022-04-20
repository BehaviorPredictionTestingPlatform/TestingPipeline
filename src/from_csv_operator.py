"""
Implements an operator for interfacing [1] to the prediction component of Pylot.

[1] https://arxiv.org/abs/2110.14870

Authors:
  * Francis Indaheng (findaheng@berkeley.edu)
  * Kaleab Belete (kaleab@berkeley.edu)
"""

import csv
import numpy as np

import erdos
from erdos import Message, Timestamp, WriteStream

from pylot.perception.detection import Obstacle
from pylot.perception.messages import ObstacleTrajectoriesMessage
from pylot.perception.tracking import ObstacleTrajectory
from pylot.utils import Location, Rotation, Transform


class FromCsvOperator(erdos.Operator):
    """Operator to stream a CSV file of historical trajectories
       to the LinearPredictorOperator of Pylot.
    
    Args:
        csv_file_path (:py:class:str): Absolute path to CSV file.
    """
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        with open(csv_file_path, 'r') as file:
            data = csv.reader(file)
            self.traj_keys = data[0]
            self.traj_data = data[1:]
        self.traj_stream = erdos.WriteStream()

    @staticmethod
    def connect():
        return [self.traj_stream]

    def run(self):
        # Dictionary for mapping agents to trajectory data
        obs_trajs = {}

        # NOTE: traj -> ['timestamp', 'agent_id', 'x', 'y', 'yaw']
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
            timestamp = Timestamp(coordinates=data['coordinates'])
            obs_traj = ObstacleTrajectory(data['obstacle'], data['trajectory'])
            msg = ObstacleTrajectoriesMessage(timestamp, obs_traj)
            self.traj_stream.send(msg)
