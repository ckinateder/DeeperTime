# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from typing import Optional
from models.modules.meta import MetaModule
import gin
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat, reduce

from models.modules.inr import INR
from models.modules.regressors import RidgeRegressor


@gin.configurable()
def deeptime(
    datetime_feats: int,
    layer_size: int,
    inr_layers: int,
    n_fourier_feats: int,
    scales: float,
    alpha: float = 0,
):
    if alpha == 0:
        print("WARNING: alpha is 0. Did you forget to set it?")
    else:
        print(f"alpha: {alpha}")
    return DeepTIMe(
        datetime_feats, layer_size, inr_layers, n_fourier_feats, scales, alpha
    )


class DeepTIMe(MetaModule):
    def __init__(
        self,
        datetime_feats: int,
        layer_size: int,
        inr_layers: int,
        n_fourier_feats: int,
        scales: float,
        alpha: float,
    ):
        super().__init__()
        self.inr = INR(
            in_feats=datetime_feats + 1,
            layers=inr_layers,
            layer_size=layer_size,
            n_fourier_feats=n_fourier_feats,
            scales=scales,
        )
        self.adaptive_weights = RidgeRegressor()

        self.alpha = alpha

        self.datetime_feats = datetime_feats
        self.inr_layers = inr_layers
        self.layer_size = layer_size
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales

    def forward(
        self, x: Tensor, x_time: Tensor, y_time: Tensor, loss: Tensor = None
    ) -> Tensor:
        """This function is the forward pass of the DeepTIMe model.
        At the end of the of the function, the ridge regressor is called to
        calculate the weights and biases, wich are then used in the forecast
        function to make predictions.

        Args:
            x (Tensor): x data
            x_time (Tensor): x time
            y_time (Tensor): y time

        Returns:
            Tensor: A tensor of predictions
        """
        # get the lookback and horizon lengths
        tgt_horizon_len = y_time.shape[1]
        batch_size, lookback_len, _ = x.shape

        # get the coords for the input to the INR and put on device
        # this is a tensor with shape (1, lookback_len + tgt_horizon_len, 1)
        coords = self.get_coords(lookback_len, tgt_horizon_len).to(x.device)

        # determine if y_time is empty
        if y_time.shape[-1] != 0:
            # concat [x_time, y_time] on the second dim
            time = torch.cat([x_time, y_time], dim=1)

            # repeat coords over time.shape
            coords = repeat(coords, "1 t 1 -> b t 1", b=time.shape[0])

            # concat the repeated coords over the last dimension
            coords = torch.cat([coords, time], dim=-1)

            # call the inr with the new coords
            time_reprs = self.inr(coords)
        else:
            # repeat the coords for each batch, pass through the INR, and
            # concatenate the time_reprs. Note that each row in the batch size
            # (the part that is repeated) is the same.
            time_reprs = repeat(
                self.inr(coords),
                "1 t d -> b t d",
                b=batch_size,
            )

        # split the time_reprs into lookback_reprs and horizon_reprs
        # recall that time_reprs has shape
        # (batch_size, lookback_len + tgt_horizon_len, layer_size)
        lookback_reprs = time_reprs[:, :-tgt_horizon_len]
        horizon_reprs = time_reprs[:, -tgt_horizon_len:]

        # calculate the weights and biases using the ridge regressor
        # this is most similar to calling the "gradient_update_parameters" in
        # the torchmeta example here (line 67):
        # https://github.com/tristandeleu/pytorch-meta/blob/master/examples/maml/train.py#L67
        weights = self.adaptive_weights(lookback_reprs, x)

        ### PUT MAML HERE ###

        # SharpMAML says y = (m + epsilon) * x + b
        # w is our parameter list
        # b is our bias list
        if loss is not None:
            e_w = self.alpha * (weights / torch.norm(weights))  # sharpmaml 8a
            weights = weights + e_w

        ### END MAML ###

        w = weights[:, :-1]
        b = weights[:, -1:]

        # make predictions using the forecast function, which just multiplies
        # the weights by the horizon_reprs and then adds the biases

        # batch_size x HORIZEN_LEN x DATASET_DIM
        preds = self.forecast(horizon_reprs, w, b)

        # return the predictions
        return preds

    def forecast(self, inp: Tensor, w: Tensor, b: Tensor) -> Tensor:
        """
        Einsum description of "... d o, ... t d -> ... t o":

        Imagine you have multiple batches of matrices (represented by the
        ellipses) where each matrix has dimensions "(d x o)" (first tensor)
        and "(t x d)" (second tensor). This einsum efficiently performs matrix
        multiplication for each batch, resulting in a new set of matrices with
        dimensions "(batch_size x t x o)".

        Effectively, this function is just a matrix multiplication between the
        weights and the input, followed by adding the biases. Classic MLP.
        """
        return torch.einsum("... d o, ... t d -> ... t o", [w, inp]) + b

    def get_coords(self, lookback_len: int, horizon_len: int) -> Tensor:
        """Get the coordinates for the input to the INR.
        This creates a single-channel "image" with the coordinates representing
        timesteps along the horizontal axis. By assigning a continuous value
        between 0 and 1 to each timestep, the model can better capture temporal
        relationships and patterns in the data.

        Args:
            lookback_len (int): how far back to look
            horizon_len (int): how far forward to look

        Returns:
            Tensor: 3d tensor with shape (1, lookback_len + horizon_len, 1)
        """

        # create equally spaced 1-D coordinates from 0 to 1 with length
        # lookback_len + horizon_len
        coords = torch.linspace(0, 1, lookback_len + horizon_len)

        # reshape the coords to into a 3D tensor with shape
        # (1, lookback_len + horizon_len, 1)
        return rearrange(coords, "t -> 1 t 1")
