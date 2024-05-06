# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from typing import Optional

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Union

from models.modules.meta import MetaModule


class RidgeRegressor(MetaModule):
    def __init__(
        self, lambda_init: Optional[float] = 0.0, alpha: Optional[float] = 0.0005
    ):
        super().__init__()
        self._lambda = nn.Parameter(torch.as_tensor(lambda_init, dtype=torch.float))
        self.alpha = alpha

    def forward(
        self,
        reprs: Tensor,
        x: Tensor,
        reg_coeff: Optional[float] = None,
        alpha: Optional[float] = None,
        loss: Optional[Tensor] = None,
    ) -> Union[Tensor, Tensor]:
        """Forward pass of the RidgeRegressor module.

        Args:
            reprs (Tensor): representation of the input
            x (Tensor): input tensor
            reg_coeff (Optional[float], optional): _description_. Defaults to None.

        Returns:
            Tensor: Weights and biases
        """
        # set reg_coeff to the default value if not provided
        if reg_coeff is None:
            reg_coeff = self.reg_coeff()
        if alpha is None:
            alpha = self.alpha

        # get weights and biases using the representation and input tensor
        w, b = self.maml_weights(reprs, x, reg_coeff, loss=loss, alpha=alpha)
        return w, b

    def get_weights(
        self, X: Tensor, Y: Tensor, reg_coeff: float
    ) -> Union[Tensor, Tensor]:
        """

        Explanation of torch.bmm(X.mT, X):
            This function is used to perform a matrix multiplication while
            reserving the batch dimension. This operation essentially
            calculates the outer product between each sample in
            X (transposed) and all other samples (including itself) across
            the batch dimension. It captures the pairwise dot products between
            samples in the feature space. X.mT is the transpose of X on the
            last two dimensions. If X has shape (batch_size, n_samples, n_dim),
            then X.mT has shape (batch_size, n_dim, n_samples). This is then
            multiplied by X, resulting in a final tensor ofshape
            (batch_size, n_dim, n_dim).


        Args:
            X (Tensor): the representation of the input
            Y (Tensor): the input tensor
            reg_coeff (float): the regularisation coefficient

        Returns:
            Tensor: _description_
        """
        # get the batch size, number of samples and number of dimensions
        batch_size, n_samples, n_dim = X.shape

        # create a column of ones and add to the input tensor for bias
        ones = torch.ones(batch_size, n_samples, 1, device=X.device)
        X = torch.concat([X, ones], dim=-1)

        # calculate the weights
        if n_samples >= n_dim:  # standard
            # if the number of samples >= number of dimensions
            # X_batch_transpose * X
            A = torch.bmm(X.mT, X)  # often called a gram matrix

            # add the regularisation coefficient to the diagonal
            # this is just getting the diagonal for each batch and adding
            # the reg_coeff to it
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)

            # X_batch_transpose * Y
            # B represents the desired output for each sample
            # based on its features.
            B = torch.bmm(X.mT, Y)

            # solve for the weights where A is the coefficients
            # and B is the desired output
            weights = torch.linalg.solve(A, B)  # AX=B, solve for X
        else:  # Woodbury
            # if the number of samples < number of dimensions
            # W = Z_transpose((Z * Z_transpose) + (reg_coeff * I))^(-1) * Y
            A = torch.bmm(X, X.mT)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            weights = torch.bmm(X.mT, torch.linalg.solve(A, Y))

        # split the weights into weights and biases
        w = weights[:, :-1]
        b = weights[:, -1:]

        return w, b

    def maml_weights(
        self,
        X: Tensor,
        Y: Tensor,
        reg_coeff: float,
        loss: float,
        alpha: float,
    ) -> Union[Tensor, Tensor]:
        """_summary_

        Args:
            X (Tensor): the representation of the input
            Y (Tensor): the input tensor
            reg_coeff (float): the regularisation coefficient

        Returns:
            Union[Tensor, Tensor]: _description_
        """
        # get the batch size, number of samples and number of dimensions
        batch_size, n_samples, n_dim = X.shape

        # create a column of ones and add to the input tensor for bias
        ones = torch.ones(batch_size, n_samples, 1, device=X.device)
        X = torch.concat([X, ones], dim=-1)

        # calculate the weights
        if n_samples >= n_dim:  # standard
            # if the number of samples >= number of dimensions
            # X_batch_transpose * X
            A = torch.bmm(X.mT, X)  # often called a gram matrix

            # add the regularisation coefficient to the diagonal
            # this is just getting the diagonal for each batch and adding
            # the reg_coeff to it
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)

            # X_batch_transpose * Y
            # B represents the desired output for each sample
            # based on its features.
            B = torch.bmm(X.mT, Y)

            # solve for the weights where A is the coefficients
            # and B is the desired output
            weights = torch.linalg.solve(A, B)  # AX=B, solve for X
        else:  # Woodbury
            # if the number of samples < number of dimensions
            # W = Z_transpose((Z * Z_transpose) + (reg_coeff * I))^(-1) * Y
            A = torch.bmm(X, X.mT)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            weights = torch.bmm(X.mT, torch.linalg.solve(A, Y))

        ### PUT MAML HERE ###
        # SharpMAML says y = (m + epsilon) * x + b
        # w is our parameter list
        # b is our bias list
        if loss is not None:
            """
            grads = torch.autograd.grad(
                loss, weights, create_graph=True, allow_unused=True
            )
            for i in range(len(grads)):
                weights[i] = torch.pow(weights[i], 2) * grads[i]
            """
            e_w = alpha * (weights / torch.norm(weights))  # sharpmaml 8a
            weights = weights + e_w

        ### END MAML ###

        # split the weights into weights and biases
        w = weights[:, :-1]
        b = weights[:, -1:]

        return w, b
        # w is basically grads = torch.autograd.grad(loss,
        # params.values(),
        # create_graph=not first_order)

    def reg_coeff(self) -> Tensor:
        return F.softplus(self._lambda)
