# -*- coding: utf-8 -*-

import os
import queue
import time
import hydra
import random
import logging
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.equations import equation_dict
from utils.data_utils import split_data
from utils.samplers import sampler_dict
from utils.reweightings import reweighting_dict
from utils.models import FullyConnectedNetwork, model_saver
from utils.plot_utils import plot_cyclic_voltammetry
from test import test2d


@hydra.main(version_base=None, config_path="./conf", config_name="Schrodinger")
def train_setup(cfg):
    log = logging.getLogger("Train")

    problem_conf = cfg["problem_conf"]
    global_conf = cfg["global_conf"]
    model_conf = cfg["model_conf"]
    train_conf = cfg["train_conf"]
    data_conf = cfg["data_conf"]
    tensorboard_writer = SummaryWriter(cfg["global_conf"]["tensorboard_path"])
    log.info(OmegaConf.to_yaml(cfg))

    # ---------------
    # global
    # ---------------
    if global_conf["seed"]:
        np.random.seed(global_conf["seed"])
        random.seed(global_conf["seed"])
        torch.manual_seed(global_conf["seed"])
        torch.cuda.manual_seed(global_conf["seed"])

    device = torch.device(global_conf["device"])
    log.info(f"device: {device}")

    # -------------
    # model
    # -------------
    log.info("create model...")
    model = FullyConnectedNetwork(model_conf)
    model.to(device)
    log.info(model)
    if model_conf.load_model:
        log.info("load weights")
        model.load_state_dict(torch.load(model_conf.model_path))
        log.info("load done...")

    # ------------
    # create data
    # ------------
    # create data_manager
    problem_define = equation_dict[cfg["name"]](problem_conf, data_conf)
    problem_define.data_generator(global_conf["seed"])  # create dataset
    log.info("create problem data successful...")

    # ---------------------
    # split training, validating, testing
    # ---------------------
    split_t_dict = {
        "train": train_conf["train_t_range"]
        # "eval": train_conf["eval_t_range"],
        # "test": train_conf["test_t_range"]
    }
    boundary_data_split_result = split_data(problem_define.boundary_data, split_t_dict, 0)
    pde_data_split_result = split_data(problem_define.pde_data, split_t_dict, 0)

    log.info("split dataset successful...")

    # ---------
    # create sampler
    # ---------
    # train data sampler
    train_initial_tensor = torch.from_numpy(problem_define.initial_data).to(device=device, dtype=torch.float)
    train_boundary_tensor = torch.from_numpy(boundary_data_split_result["train"]).to(device=device, dtype=torch.float)
    train_pde_tensor = torch.from_numpy(pde_data_split_result["train"]).to(device=device, dtype=torch.float)
    train_pde_tensor.requires_grad = True
    if problem_conf["boundary_cond"] == "periodic" or problem_conf["boundary_cond"] == "Neumann":
        train_boundary_tensor.requires_grad = True

    train_pde_sampler = sampler_dict[train_conf["pde_sampler"]](
        train_pde_tensor, reweighting_dict[train_conf["pde_reweighting"]](train_conf["reweighting_params"]),
        model=model,
        loss_func=problem_define.compute_loss_basic_weights,
        **train_conf["sampler_conf"]
    )
    train_initial_sampler = sampler_dict["UniformSampler"](train_initial_tensor, reweighting_dict["NoReWeighting"]())
    train_boundary_sampler = sampler_dict["UniformSampler"](train_boundary_tensor, reweighting_dict["NoReWeighting"]())

    # test data
    # data_path = "E:/马金博/code/SolvingPDE/data/test.csv"
    project_root = get_original_cwd()
    test_data = pd.read_csv('{}/data/test.csv'.format(project_root))

    test_input_t = torch.from_numpy(np.array(test_data['t'])).float().to(device)
    test_input_x = torch.from_numpy(np.array(test_data['x'])).float().to(device)
    test_input_tensor = torch.stack([test_input_t, test_input_x], dim=1)

    # -------------
    # optimizer
    # -------------
    if train_conf["optim"] == "adam":
        optim = torch.optim.Adam(model.parameters(), **train_conf["optim_conf"])
    elif train_conf["optim"] == "sgd":
        optim = torch.optim.SGD(model.parameters(), **train_conf["optim_conf"])
    else:
        raise ValueError("optim: {} is not supported".format(train_conf["optim"]))
    if train_conf["sch"] == "step":
        step_sch_conf = train_conf["sch_step"]
        sch = torch.optim.lr_scheduler.StepLR(optim, step_size=int(
            train_conf["main_conf"]["max_steps"] / step_sch_conf["stage"]),
                                              gamma=step_sch_conf["gamma"])
    elif train_conf["sch"] == "cos":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim, train_conf["main_conf"]["max_steps"],
                                                         train_conf["optim_conf"]["lr"] / 30)
    elif train_conf["sch"] == "expon":
        sch = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
    elif train_conf["sch"] == "onecycle":
        sch = torch.optim.lr_scheduler.OneCycleLR(optim,
                                                  max_lr=train_conf["optim_conf"]["lr"],
                                                  pct_start=train_conf["sch_par"]["pct_start"],
                                                  total_steps=train_conf["main_conf"]["max_steps"])

    elif train_conf["sch"] == "plateau":
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=10, factor=0.8,
                                                         verbose=False, min_lr=1e-8)

    elif train_conf["sch"] == "cyclic":
        # only support SGD
        base_lr = train_conf["optim_conf"]["lr"] / 10
        max_lr = train_conf["optim_conf"]["lr"]
        sch = torch.optim.lr_scheduler.CyclicLR(optim,
                                                base_lr=base_lr,
                                                max_lr=max_lr,
                                                step_size_up=int(train_conf["main_conf"]["max_steps"] / 2),
                                                mode='triangular')
    elif train_conf["sch"] == "Identify":
        sch = None
    else:
        raise ValueError("sch: {} is not supported".format(train_conf["sch"]))

    # -------------
    # main loop
    # -------------
    best_eval_loss = 1e6
    best_model_save_path = None
    train_main_conf = train_conf["main_conf"]
    model_save_queue = queue.Queue(maxsize=5)
    for step in range(train_main_conf["max_steps"]):

        train_pde_data = train_pde_sampler.sampler(train_main_conf["pde_batch_size"])
        train_initial_data = train_initial_sampler.sampler(train_main_conf["initial_batch_size"])
        train_boundary_data = train_boundary_sampler.sampler(train_main_conf["boundary_batch_size"])

        optim.zero_grad()
        loss_dict = problem_define.compute_loss(model, train_pde_data, train_initial_data, train_boundary_data, "train")
        optim.step()
        if sch is not None:
            sch.step()

        if step % train_main_conf["print_frequency"] == 0:
            log.info(f"step: {step}")
            for key, value in loss_dict.items():
                log.info("{} loss: {:.5e}".format(key, value))
                tensorboard_writer.add_scalar(f"TrainLoss/{key}", value, step)
            log.info("learning rate: {}".format(optim.param_groups[0]['lr']))
            tensorboard_writer.add_scalar(f"learning-rate", optim.param_groups[0]['lr'], step)

        if step % train_main_conf["eval_frequency"] == 0:
            log.info("save model")

            if best_eval_loss > loss_dict["total"]:
                best_eval_loss = loss_dict["total"]
                best_model_save_path = model_saver(
                    save_folder=train_main_conf["model_save_folder"],
                    model=model,
                    save_name=train_main_conf["model_basic_save_name"],
                    step=step
                )

                if model_save_queue.full():
                    del_step = model_save_queue.get()
                    del_path = os.path.join(train_main_conf["model_save_folder"],
                                            "{}_{}.pth".format(train_main_conf["model_basic_save_name"], del_step))
                    os.remove(del_path)

                model_save_queue.put(step)

            model.train()
    log.info("best loss: {:.5e}".format(best_eval_loss))
    log.info("train done...")

    # ---------
    # testing
    # ---------
    log.info("begin test...")
    log.info("{}".format(best_model_save_path))
    model.load_state_dict(torch.load(best_model_save_path))

    model.eval()

    with torch.no_grad():
        eval_pred = model(test_input_tensor.to(device)).detach()
        eval_pred = torch.sqrt(eval_pred[:, 0:1] ** 2 + eval_pred[:, 1:2] ** 2).cpu().numpy()
    eval_pred = eval_pred.reshape(-1)
    # 输出到csv
    df = pd.DataFrame()  # 创建DataFrame
    df["id"] = range(test_data['t'].shape[0])  # 创建id列
    df["t"] = test_data['t']  # 创建t列
    df["x"] = test_data['x']  # 创建x列
    df["pred"] = eval_pred  # 创建pred列
    df.to_csv("submission.csv", index=False)

    log.info("test done...")


if __name__ == "__main__":
    train_setup()
