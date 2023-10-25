# SolvingPDE
2023组内流光杯比赛--人工智能驱动偏微分方程求解

# Env

* python=3.8
* torch=1.13
* hydra-core==1.2
* tensorboard==2.14.0
* matplotlib==3.7.3
* scipy==1.10.1
* sympy==1.12
* pandas==2.0.3

# TODO List

- [x] baseline
- [x] 基础数据采样器
- [x] 基础损失函数
- [x] 基础参数配置
- [x] 完整训练及推理流程
- [ ] 第一轮调试超参：
- [ ] 训练轮数-steps=100000、500000、1000000
- [ ] 数据采集数量
- [ ] 网络深度
- [ ] 网络宽度
- [ ] 学习率调整
- [ ] 模型选择：
- [ ] cPINN
- [ ] xPINN
- [ ] 采样方式：
- [ ] DMIS采样
- [ ] NDMIS采样
- [ ] ......


# Training Tasks

- [x] Baseline：python train.py --config-name=Schrodinger hydra.job.chdir=True
- [x] step=100000：python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=100000 hydra.job.chdir=True
- [x] step=500000：python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 hydra.job.chdir=True
- [ ] step=1000000：python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=1000000 hydra.job.chdir=True

# Results

|        Model        |  Loss_Total  |  Score   |       model        |
|:-------------------:|:------------:|:--------:|:------------------:|
| baseline-step=20000 | 2.63425e-03  | 0.18794  |                    |
|     step=100000     | 8.09942e-05  | 0.03687  | Schrodinger_99500  |
|     step=500000     | 1.43624e-05  | 0.01810  | Schrodinger_483500 |
|    step=1000000     | 9.23105e-06  | 0.01728  | Schrodinger_907000 | 