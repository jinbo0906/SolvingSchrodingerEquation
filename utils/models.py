# -*- coding: utf-8 -*-

import os
from numbers import Number
import torch
from torch import nn
import torch.nn.functional as F


class AdaptiveTanh(nn.Module):

    def __init__(self, scale_factor=1.0):
        super(AdaptiveTanh, self).__init__()

        self.param = nn.Parameter(torch.tensor(1.0))
        self.activate_func = nn.Tanh()
        self.scale_factor = scale_factor

    def forward(self, x):
        return self.activate_func(self.param * x * self.scale_factor)


class SiLU(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)


class FullyConnectedNetwork(nn.Module):

    def __init__(self, conf):
        super(FullyConnectedNetwork, self).__init__()

        self.layer_conf = conf["layer"]
        self.dim_conf = conf["dim"]

        if isinstance(self.layer_conf["layer_size"], Number):
            layer_size = [self.layer_conf["layer_size"]] * self.layer_conf["layer_n"]
        else:
            layer_size = self.layer_conf["layer_size"]
            assert len(layer_size) == self.layer_conf["layer_n"]

        self._network = nn.Sequential()
        curr_dim = self.dim_conf["input_dim"]
        activate_func = self.layer_conf["activate"]
        norm_flag = self.layer_conf["norm"]

        for layer_id, layer_dim in enumerate(layer_size):
            self._network.add_module(
                "layer_{}".format(layer_id + 1),
                self._make_layer(curr_dim, layer_dim, norm_flag, activate_func, self.layer_conf["activate_scale_factor"])
            )
            curr_dim = layer_dim

        self._network.add_module(
            "layer_{}".format(len(layer_size) + 1),
            self._make_layer(curr_dim,
                             self.dim_conf["output_dim"],
                             activate_func=self.layer_conf["final_activate"],
                             activate_scale_factor=self.layer_conf["activate_scale_factor"])
        )

    def _forward_impl(self, x):
        return self._network(x)

    def forward(self, x):
        return self._forward_impl(x)

    @staticmethod
    def _make_layer(input_dim, output_dim, norm=False, activate_func="tanh", activate_scale_factor=1.0):
        layers = list()

        layers.append(
            nn.Linear(input_dim, output_dim)
        )

        if norm:
            layers.append(
                nn.BatchNorm1d(output_dim)
            )

        if activate_func == "tanh":
            layers.append(nn.Tanh())
        elif activate_func == "adaptive_tanh":
            layers.append(AdaptiveTanh(activate_scale_factor))
        elif activate_func == "silu":
            layers.append(SiLU())
        elif activate_func == "Identify":
            pass
        else:
            raise ValueError

        return nn.Sequential(*layers)


def model_saver(save_folder, model, save_name, step=None):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if step is not None:
        save_path = os.path.join(save_folder, "{}_{}.pth".format(save_name, step))
    else:
        save_path = os.path.join(save_folder, "{}.pth".format(save_name))
    torch.save(model.state_dict(), save_path)
    return save_path
