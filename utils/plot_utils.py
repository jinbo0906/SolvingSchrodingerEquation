# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def interpolate_2d(invar, outvar, plotsize_x=100, plotsize_y=100):

    # create grid
    extent = (invar[:, 0].min(), invar[:, 0].max(), invar[:, 1].min(), invar[:, 1].max())
    _plot_mesh = np.meshgrid(
        np.linspace(extent[0], extent[1], plotsize_x),
        np.linspace(extent[2], extent[3], plotsize_y),
        indexing="ij"
    )

    outvar_interp = griddata(
        invar, outvar, tuple(_plot_mesh)
    )
    return extent, outvar_interp, _plot_mesh


def mesh_plotter_2d(coords, simplices, step=None, ex_path="./mesh_data", name="mesh"):
    """
    function to plot triangular meshes
    """
    assert coords.shape[1] == 2
    plt.figure(figsize=(20, 20), dpi=100)

    plt.triplot(coords[:, 0], coords[:, 1], simplices)

    if step is not None:
        plt.savefig(os.path.join(ex_path, "{}_{}.png".format(name, step)))
    else:
        plt.savefig(os.path.join(ex_path, "{}.png".format(name)))
    plt.close()


def plot_cyclic_voltammetry(model, output_path, ground_true, sigma, t_max, x_max, theta_init, theta_switch, empirical_model, if_plot=False):
    """
    plot cyclic voltammetry
    :param model: PINNs model
    :param output_path: output path
    :param ground_true: numerical results
    :param sigma: dimensionless scan rate
    :param t_max: max dimensionless time
    :param x_max: max dimensionless spatial coordinates
    :param theta_init: init voltage
    :param theta_switch: reverse voltage
    :param empirical_model: Randles-Sevcik or Hubbard
    :param if_plot: False
    :return: None
    """

    t_flat = np.linspace(0, t_max, ground_true.shape[0])
    x_flat = np.linspace(0, x_max, ground_true.shape[1])

    with torch.no_grad():

        input_t = torch.from_numpy(t_flat.reshape(-1, 1)).to(device=torch.device("cuda"), dtype=torch.float)
        input_x0 = torch.from_numpy(np.ones_like(t_flat).reshape(-1, 1) * x_flat[0]).to(device=torch.device("cuda"), dtype=torch.float)
        input_x1 = torch.from_numpy(np.ones_like(t_flat).reshape(-1, 1) * x_flat[1]).to(device=torch.device("cuda"), dtype=torch.float)

        pred_x0 = model(torch.concat([input_t, input_x0], dim=1)).cpu().numpy()
        pred_x1 = model(torch.concat([input_t, input_x1], dim=1)).cpu().numpy()

    plt.figure()
    plt.xlabel("Potential")
    plt.ylabel("flux")

    # draw cyclic_voltammetry with numerical method
    cv_flat = np.where(t_flat < t_max/2.0, theta_init-sigma*t_flat, theta_switch+sigma*(t_flat-t_max/2.0))
    plt.plot(cv_flat, -(ground_true[:, 1] - ground_true[:, 0])[::-1] / x_flat[1], c="b", label="Numerical")

    # draw cyclic_voltammetry with pinns
    plt.plot(cv_flat, -(pred_x1 - pred_x0)[::-1] / x_flat[1], c="r", label="PINN")

    # print(np.min(-(ground_true[:, 1] - ground_true[:, 0])[::-1] / x_flat[1]))
    # print(np.min(-(pred_x1 - pred_x0)[::-1] / x_flat[1]))

    # draw results of empirical model
    if empirical_model == "Randles-Sevcik":

        plt.axhline(-0.446 * np.sqrt(sigma), label='R-S equation', ls='-.', color='black')
        plt.axvline(-1.109, label='Expected Forward Scan Potential', ls='--', color='k')

        # x_list = np.linspace(theta_switch/2, theta_init/2, 20)
        # y_list = np.ones_like(x_list) * (-0.4463 * np.sqrt(sigma))
        # plt.plot(x_list, y_list, c="black", label="Randles-Sevcik")
    elif empirical_model == "Hubbard":
        # TODO prediction of Hubbard model
        pass
    else:
        raise KeyError

    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_path, "cyclic_voltammetry.png"))

    if if_plot:
        plt.show()
    else:
        plt.clf()
        plt.close()

