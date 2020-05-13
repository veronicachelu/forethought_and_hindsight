import os
from typing import Any
from typing import Callable, Sequence, Tuple

import dm_env
import numpy as np
import tensorflow as tf
from dm_env import specs
from jax import numpy as jnp

from agents.tabular.PAML.tp_bw_PAML import TpBwPAML
from utils.replay import Replay

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class TpBwProjPAML(TpBwPAML):
    def __init__(
            self,
            **kwargs
    ):

        super(TpBwProjPAML, self).__init__(**kwargs)

        def project(params):
            o_norm = np.linalg.norm(np.asarray(params), ord=2)
            params = np.divide(params, o_norm, out=np.zeros_like(params), where=o_norm != 0)
            params *= self._max_norm
            return params

        self._model_opt_update = lambda gradient, param: param + self._lr_model * gradient
        self._projected_model_opt_update = lambda gradient, param: project(param + self._lr_model * gradient)

    def model_update(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        if self._n == 0:
            return
        if len(self._sequence) >= self._n:
            o_tmn = self._sequence[0][0]
            o_t = self._sequence[-1][-1]
            losses, gradients = self._model_loss_grad(self._v_network, self._o_network, self._r_network, self._sequence)

            self._o_network[o_t] = \
                self._projected_model_opt_update(gradients[0], self._o_network[o_t])
            self._r_network[o_tmn, o_t] = \
                self._model_opt_update(gradients[1], self._r_network[o_tmn, o_t])
            total_loss, o_loss, r_loss = losses
            o_grad, r_grad = gradients
            o_grad = np.linalg.norm(np.asarray(o_grad), ord=2)
            o_norm = np.max(np.linalg.norm(np.asarray(self._o_network), ord=2, axis=-1))
            losses_and_grads = {"losses": {
                "loss": total_loss,
                "o_loss": o_loss,
                "r_loss": r_loss,
                "o_grad": o_grad,
                "o_norm": o_norm,
                "r_grad": r_grad,
            },
            }
            print("max_param_norm {}".format(o_norm))
            self._log_summaries(losses_and_grads, "model")
            self._sequence = self._sequence[1:]

        if self._should_reset_sequence:
            self._sequence = []
            self._should_reset_sequence = False