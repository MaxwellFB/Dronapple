"""
Suction system from the Component 3
"""

import pathlib as pl

from joblib import load as joblib_load
from torch import load, Tensor


class Suction:
    def __init__(self, path_model='suction_network.pth', path_scaler='suction_std_scaler.bin'):
        root_dir = pl.Path(__file__).resolve().parent
        self.model = load(pl.Path(root_dir, path_model))
        self.scaler = joblib_load(pl.Path(root_dir, path_scaler))

    def predict(self, input_data):
        """
        Predict if is to grab the apple or not

        Args:
            input_data (tuple[float, float, float, float, float, float, float]):
                output_boxes: Coordinates bounding box (X, Y, W, H), normalized between 0.0 and 1.0
                drone_velocity: Current drone velocity (X, Y, Z), values between -1.0 and 1.0

        Returns:
             bool: Tag to indicate whether to grab the apple
        """
        # Code configured to run only in CPU
        input_scaled = self.scaler.transform([input_data])
        input_scaled = Tensor(input_scaled)
        pred_sigmoid = self.model(input_scaled)
        pred = pred_sigmoid.detach().numpy() > 0.5
        return pred[0][0]
