"""
Implements an operator for interfacing the Pylot LinearPredictorOperator to [1].

[1] https://arxiv.org/abs/2110.14870

Authors:
  * Francis Indaheng (findaheng@berkeley.edu)
"""

import csv
import erdos


class ToCsvOperator(erdos.Operator):
    """Operator to write the prediction stream of
    the Pylot LinearPredictorOperator to a CSV file.
    
    Args:
        prediction_stream (:py:class:`erdos.ReadStream`): Stream on which messages
            (:py:class:`pylot.prediction.messages.PredictionMessage`) are received.
        csv_file_dir (:py:class:str): Absolute dir path to write CSV file.
        worker_num (:py:class:int): Parallel worker identifier.
    """
    def __init__(self, prediction_stream: erdos.ReadStream, csv_file_dir: str, worker_num: int):
        self.prediction_stream = prediction_stream
        self.csv_file = open(f'{csv_file_dir}/pred_{worker_num}_0.csv', 'w', newline='')
        self.writer = csv.writer(self.csv_file)

    @staticmethod
    def connect(prediction_stream: erdos.ReadStream):
        return []

    def destroy(self):
        self.csv_file.close()

    def run(self):
        self.writer.writerow(['timestamp', 'agent_id', 'x', 'y', 'yaw'])
        while (msg := self.prediction_stream.try_read() is not None):
            timestamp = msg.timestamp.coordinates[0]
            for pred in msg.predictions:
                agent_id = pred.obstacle_trajectory.obstacle.id
                for traj in pred.predicted_trajectory:
                    x, y = traj.location.x, traj.location.y
                    yaw = traj.rotation.yaw
                    self.writer.writerow([timestamp, agent_id, x, y, yaw])
