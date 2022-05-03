"""
Implements an operator for interfacing a Pylot prediction operator to [1].

[1] https://arxiv.org/abs/2110.14870

Authors:
  * Francis Indaheng (findaheng@berkeley.edu)
"""

import csv
import erdos


class ToCsvOperator(erdos.Operator):
    """Operator to write the prediction stream of
    a Pylot prediction operator to a CSV file.
    
    Args:
        prediction_stream (:py:class:`erdos.ReadStream`): Stream on which messages
            (:py:class:`pylot.prediction.messages.PredictionMessage`) are received.
        csv_file_dir (:py:class:str): Absolute dir path to write CSV file.
        worker_num (:py:class:int): Parallel worker identifier.
    """
    def __init__(self, prediction_stream: erdos.ReadStream, csv_file_dir: str, worker_num: int):
        prediction_stream.add_callback(self.write_to_csv, [])
        self._logger = erdos.utils.setup_logging(self.config.name, self.config.log_file_name)
        self.csv_file = open(f'{csv_file_dir}/pred_{worker_num}.csv', 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['timestamp', 'agent_id', 'x', 'y', 'yaw'])

    @staticmethod
    def connect(prediction_stream: erdos.ReadStream):
        return []

    def destroy(self):
        self._logger.warn(f'Destroying {self.config.name}')
        self.csv_file.close()

    @erdos.profile_method()
    def write_to_csv(self, msg: erdos.Message):
        for pred in msg.predictions:
            agent_id = pred.obstacle_trajectory.obstacle.id
            for i, traj in enumerate(pred.predicted_trajectory):
                timestamp = msg.timestamp.coordinates[i]
                x, y = traj.location.x, traj.location.y
                yaw = traj.rotation.yaw
                self.writer.writerow([timestamp, agent_id, x, y, yaw])
