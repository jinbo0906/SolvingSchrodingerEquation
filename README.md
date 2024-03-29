# SolvingSchrodingerEquation
2023组内流光杯初赛--人工智能驱动薛定谔方程求解

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
- [x] 训练轮数-steps=100000、500000、1000000
- [x] 数据采集数量-pde_data_n=2000、4000、10000、20000、40000、60000、80000
- [x] 网络深度
- [x] 网络宽度
- [x] 第二轮调试超参：
- [x] 学习率调整
- [x] 自适应激活函数
- [ ] 采样方式：
- [ ] DMIS采样
- [ ] 模型选择：
- [ ] cPINN
- [ ] xPINN
- [ ] gPINN
- [ ] ......


# Training Tasks

- [x] Baseline：python train.py --config-name=Schrodinger hydra.job.chdir=True
- [x] step=100000：python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=100000 hydra.job.chdir=True
- [x] step=500000：python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 hydra.job.chdir=True
- [x] step=1000000：python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=1000000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=40000：CUDA_VISIBLE_DEVICES=0 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=40000 data_conf.initial_data_n=200 data_conf.boundary_data_n=200 hydra.job.chdir=True
- [x] step=500000,pde_data_n=80000：CUDA_VISIBLE_DEVICES=1 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=200 data_conf.boundary_data_n=200 hydra.job.chdir=True
- [x] step=500000,pde_data_n=100000：CUDA_VISIBLE_DEVICES=2 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=100000 data_conf.initial_data_n=200 data_conf.boundary_data_n=200 hydra.job.chdir=True
- [x] step=500000,pde_data_n=150000：CUDA_VISIBLE_DEVICES=3 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=150000 data_conf.initial_data_n=200 data_conf.boundary_data_n=200 hydra.job.chdir=True
- [x] step=500000,pde_data_n=200000：CUDA_VISIBLE_DEVICES=0 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=200000 data_conf.initial_data_n=200 data_conf.boundary_data_n=200 hydra.job.chdir=True
- [x] step=500000,pde_data_n=250000：CUDA_VISIBLE_DEVICES=1 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=250000 data_conf.initial_data_n=200 data_conf.boundary_data_n=200 hydra.job.chdir=True
- [x] pde_data_n=2000: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=100000 data_conf.pde_data_n=2000 hydra.job.chdir=True
- [x] pde_data_n=4000: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=100000 data_conf.pde_data_n=4000 hydra.job.chdir=True
- [x] pde_data_n=10000: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=100000 data_conf.pde_data_n=10000 hydra.job.chdir=True
- [x] pde_data_n=20000: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=100000 data_conf.pde_data_n=20000 hydra.job.chdir=True
- [x] pde_data_n=40000: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=100000 data_conf.pde_data_n=40000 hydra.job.chdir=True
- [x] pde_data_n=60000: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=100000 data_conf.pde_data_n=60000 hydra.job.chdir=True
- [x] pde_data_n=80000: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=100000 data_conf.pde_data_n=80000 hydra.job.chdir=True
- [x] pde_data_n=100000: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=100000 data_conf.pde_data_n=100000 hydra.job.chdir=True
- [x] pde_data_n=150000: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=100000 data_conf.pde_data_n=150000 hydra.job.chdir=True
- [x] pde_data_n=80000,in,bo=500: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=100000 data_conf.pde_data_n=80000 data_conf.initial_data_n=500 data_conf.boundary_data_n=500 hydra.job.chdir=True
- [x] pde_data_n=80000,in,bo=1000: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=100000 data_conf.pde_data_n=80000 data_conf.initial_data_n=1000 data_conf.boundary_data_n=1000 hydra.job.chdir=True
- [x] pde_data_n=80000,in,bo=3000: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=100000 data_conf.pde_data_n=80000 data_conf.initial_data_n=3000 data_conf.boundary_data_n=3000 hydra.job.chdir=True
- [x] pde_data_n=80000,in,bo=5000: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=100000 data_conf.pde_data_n=80000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 hydra.job.chdir=True
- [x] pde_data_n=40000,in,bo=500: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=100000 data_conf.pde_data_n=40000 data_conf.initial_data_n=500 data_conf.boundary_data_n=500 hydra.job.chdir=True
- [x] pde_data_n=40000,in,bo=1000: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=100000 data_conf.pde_data_n=40000 data_conf.initial_data_n=1000 data_conf.boundary_data_n=1000 hydra.job.chdir=True
- [x] pde_data_n=40000,in,bo=3000: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=100000 data_conf.pde_data_n=40000 data_conf.initial_data_n=3000 data_conf.boundary_data_n=3000 hydra.job.chdir=True
- [ ] pde_data_n=40000,in,bo=5000: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=100000 data_conf.pde_data_n=40000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=40000,in,bo=3000: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=40000 data_conf.initial_data_n=3000 data_conf.boundary_data_n=3000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=80000,in,bo=1000: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=1000 data_conf.boundary_data_n=1000 hydra.job.chdir=True
- [x] step=300000,layer_size=100: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=300000 model_conf.layer.layer_size=100 hydra.job.chdir=True
- [x] step=300000,layer_size=200: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=300000 model_conf.layer.layer_size=200 hydra.job.chdir=True
- [x] step=300000,layer_size=100,layer_n=3: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=300000 model_conf.layer.layer_n=3 model_conf.layer.layer_size=100 hydra.job.chdir=True
- [x] step=500000,layer_size=100,layer_n=3: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=3 model_conf.layer.layer_size=100 hydra.job.chdir=True
- [x] step=500000,layer_size=200,layer_n=3: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=3 model_conf.layer.layer_size=200 hydra.job.chdir=True
- [x] step=300000,layer_size=64,layer_n=3: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=300000 model_conf.layer.layer_n=3 model_conf.layer.layer_size=64 hydra.job.chdir=True
- [x] step=500000,layer_size=100,layer_n=4: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=4 model_conf.layer.layer_size=100 data_conf.pde_data_n=80000 data_conf.initial_data_n=1000 data_conf.boundary_data_n=1000 hydra.job.chdir=True
- [x] step=500000,layer_size=64,layer_n=3: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=3 model_conf.layer.layer_size=64 data_conf.pde_data_n=80000 data_conf.initial_data_n=1000 data_conf.boundary_data_n=1000 hydra.job.chdir=True
- [x] step=500000,layer_size=50,layer_n=3: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=3 model_conf.layer.layer_size=50 data_conf.pde_data_n=80000 data_conf.initial_data_n=1000 data_conf.boundary_data_n=1000 hydra.job.chdir=True
- [x] step=500000,layer_size=40,layer_n=3: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=3 model_conf.layer.layer_size=40 data_conf.pde_data_n=80000 data_conf.initial_data_n=1000 data_conf.boundary_data_n=1000 hydra.job.chdir=True
- [x] step=500000,layer_size=100,layer_n=3: python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=3 model_conf.layer.layer_size=100 data_conf.pde_data_n=80000 data_conf.initial_data_n=1000 data_conf.boundary_data_n=1000 hydra.job.chdir=True
- [x] step=500000,layer_size=200,layer_n=3: CUDA_VISIBLE_DEVICES=0 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=3 model_conf.layer.layer_size=200 data_conf.pde_data_n=80000 data_conf.initial_data_n=1000 data_conf.boundary_data_n=1000 hydra.job.chdir=True
- [x] step=500000,layer_n=3(1080Ti): CUDA_VISIBLE_DEVICES=0 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=3 model_conf.layer.layer_size=30,40,50,64,100,200 data_conf.pde_data_n=80000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 hydra.job.chdir=True
- [x] step=500000,layer_n=4(1080Ti): CUDA_VISIBLE_DEVICES=0 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=4 model_conf.layer.layer_size=30,40,50,64,100,200 data_conf.pde_data_n=80000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 hydra.job.chdir=True
- [x] step=500000,layer_n=2(1080Ti): CUDA_VISIBLE_DEVICES=0 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=2 model_conf.layer.layer_size=30,40,50,64,100,200 data_conf.pde_data_n=80000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 hydra.job.chdir=True
- [x] step=500000,layer_n=2(1080Ti): CUDA_VISIBLE_DEVICES=0 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=2 model_conf.layer.layer_size=40,50,64,100,200 data_conf.pde_data_n=80000 data_conf.initial_data_n=1000 data_conf.boundary_data_n=1000 hydra.job.chdir=True
- [x] step=500000,layer_n=2(1080Ti): CUDA_VISIBLE_DEVICES=1 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=2 model_conf.layer.layer_size=300,400,500,600 data_conf.pde_data_n=80000 data_conf.initial_data_n=1000 data_conf.boundary_data_n=1000 hydra.job.chdir=True
- [x] step=500000,layer_n=2(1080Ti): CUDA_VISIBLE_DEVICES=1 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=2 model_conf.layer.layer_size=300,400,500,600 data_conf.pde_data_n=80000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 hydra.job.chdir=True
- [x] step=1000000,layer_n=2(1080Ti): CUDA_VISIBLE_DEVICES=2 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=1000000 model_conf.layer.layer_n=2 model_conf.layer.layer_size=500 data_conf.pde_data_n=80000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 hydra.job.chdir=True
- [x] step=500000,layer_n=4(1080Ti): CUDA_VISIBLE_DEVICES=0 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=4 model_conf.layer.layer_size=64 data_conf.pde_data_n=80000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 train_conf.main_conf.initial_batch_size=100 train_conf.main_conf.boundary_batch_size=100 hydra.job.chdir=True
- [x] step=500000,layer_n=4(1080Ti): CUDA_VISIBLE_DEVICES=1 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=4 model_conf.layer.layer_size=64 data_conf.pde_data_n=80000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 train_conf.main_conf.initial_batch_size=200 train_conf.main_conf.boundary_batch_size=200 hydra.job.chdir=True
- [x] step=500000,layer_n=4(1080Ti): CUDA_VISIBLE_DEVICES=0 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=4 model_conf.layer.layer_size=64 data_conf.pde_data_n=80000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 train_conf.main_conf.initial_batch_size=500 train_conf.main_conf.boundary_batch_size=500 hydra.job.chdir=True
- [x] step=500000,layer_n=4(1080Ti): CUDA_VISIBLE_DEVICES=1 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=4 model_conf.layer.layer_size=64 data_conf.pde_data_n=80000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 train_conf.main_conf.initial_batch_size=1000 train_conf.main_conf.boundary_batch_size=1000 hydra.job.chdir=True
- [x] step=500000,layer_n=4(1080Ti): CUDA_VISIBLE_DEVICES=0 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=4 model_conf.layer.layer_size=64 data_conf.pde_data_n=80000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 train_conf.main_conf.initial_batch_size=2000 train_conf.main_conf.boundary_batch_size=2000 hydra.job.chdir=True
- [x] step=500000,layer_n=4(1080Ti): CUDA_VISIBLE_DEVICES=0 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=4 model_conf.layer.layer_size=64 data_conf.pde_data_n=80000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 train_conf.main_conf.pde_batch_size=10000,30000 hydra.job.chdir=True
- [x] step=500000,layer_n=4(1080Ti): CUDA_VISIBLE_DEVICES=0 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=4 model_conf.layer.layer_size=64 data_conf.pde_data_n=80000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 train_conf.main_conf.pde_batch_size=40000,50000 hydra.job.chdir=True
- [x] step=500000,layer_n=4(1080Ti): CUDA_VISIBLE_DEVICES=1 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=4 model_conf.layer.layer_size=64 data_conf.pde_data_n=80000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 train_conf.main_conf.pde_batch_size=10000,30000 train_conf.main_conf.initial_batch_size=1000 train_conf.main_conf.boundary_batch_size=1000 hydra.job.chdir=True
- [x] step=500000,layer_n=4(1080Ti): CUDA_VISIBLE_DEVICES=1 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=4 model_conf.layer.layer_size=64 data_conf.pde_data_n=80000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 train_conf.main_conf.pde_batch_size=40000,50000 train_conf.main_conf.initial_batch_size=1000 train_conf.main_conf.boundary_batch_size=1000 hydra.job.chdir=True
- [x] step=500000,layer_n=4(1080Ti): CUDA_VISIBLE_DEVICES=6 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=4 model_conf.layer.layer_size=64 data_conf.pde_data_n=80000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 train_conf.main_conf.pde_batch_size=80000 train_conf.main_conf.initial_batch_size=50 train_conf.main_conf.boundary_batch_size=50 hydra.job.chdir=True
- [x] step=500000,layer_n=4(1080Ti): CUDA_VISIBLE_DEVICES=4 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=4 model_conf.layer.layer_size=64 data_conf.pde_data_n=80000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 train_conf.main_conf.pde_batch_size=80000 train_conf.main_conf.initial_batch_size=1000 train_conf.main_conf.boundary_batch_size=1000 hydra.job.chdir=True
- [x] step=500000,layer_n=4(1080Ti): CUDA_VISIBLE_DEVICES=5 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 model_conf.layer.layer_n=4 model_conf.layer.layer_size=64 data_conf.pde_data_n=80000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 train_conf.main_conf.pde_batch_size=80000 train_conf.main_conf.initial_batch_size=5000 train_conf.main_conf.boundary_batch_size=5000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=40000,in,bo=500：CUDA_VISIBLE_DEVICES=0 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=40000 data_conf.initial_data_n=500 data_conf.boundary_data_n=500 hydra.job.chdir=True
- [x] step=500000,pde_data_n=40000,in,bo=1000：CUDA_VISIBLE_DEVICES=1 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=40000 data_conf.initial_data_n=1000 data_conf.boundary_data_n=1000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=40000,in,bo=5000：CUDA_VISIBLE_DEVICES=2 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=40000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=80000,in,bo=500：CUDA_VISIBLE_DEVICES=3 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=500 data_conf.boundary_data_n=500 hydra.job.chdir=True
- [x] step=500000,pde_data_n=80000,in,bo=3000：CUDA_VISIBLE_DEVICES=0 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=3000 data_conf.boundary_data_n=3000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=80000,in,bo=5000：CUDA_VISIBLE_DEVICES=1 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=150000,in,bo=500：CUDA_VISIBLE_DEVICES=2 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=150000 data_conf.initial_data_n=500 data_conf.boundary_data_n=500 hydra.job.chdir=True
- [x] step=500000,pde_data_n=150000,in,bo=1000：CUDA_VISIBLE_DEVICES=3 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=150000 data_conf.initial_data_n=1000 data_conf.boundary_data_n=1000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=150000,in,bo=3000：CUDA_VISIBLE_DEVICES=4 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=150000 data_conf.initial_data_n=3000 data_conf.boundary_data_n=3000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=150000,in,bo=5000：CUDA_VISIBLE_DEVICES=4 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=150000 data_conf.initial_data_n=5000 data_conf.boundary_data_n=5000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=40000,in,bo=8000：CUDA_VISIBLE_DEVICES=5 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=40000 data_conf.initial_data_n=8000 data_conf.boundary_data_n=8000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=40000,in,bo=10000：CUDA_VISIBLE_DEVICES=5 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=40000 data_conf.initial_data_n=10000 data_conf.boundary_data_n=10000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=80000,in,bo=8000：CUDA_VISIBLE_DEVICES=6 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=8000 data_conf.boundary_data_n=8000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=80000,in,bo=10000：CUDA_VISIBLE_DEVICES=6 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=10000 data_conf.boundary_data_n=10000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=150000,in,bo=8000：CUDA_VISIBLE_DEVICES=7 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=150000 data_conf.initial_data_n=8000 data_conf.boundary_data_n=8000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=150000,in,bo=10000：CUDA_VISIBLE_DEVICES=7 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=150000 data_conf.initial_data_n=10000 data_conf.boundary_data_n=10000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=40000,in,bo=15000：CUDA_VISIBLE_DEVICES=0 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=40000 data_conf.initial_data_n=15000 data_conf.boundary_data_n=15000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=80000,in,bo=15000：CUDA_VISIBLE_DEVICES=1 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=15000 data_conf.boundary_data_n=15000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=150000,in,bo=15000：CUDA_VISIBLE_DEVICES=2 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=150000 data_conf.initial_data_n=15000 data_conf.boundary_data_n=15000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=40000,in,bo=20000：CUDA_VISIBLE_DEVICES=3 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=40000 data_conf.initial_data_n=20000 data_conf.boundary_data_n=20000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=80000,in,bo=20000：CUDA_VISIBLE_DEVICES=4 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=20000 data_conf.boundary_data_n=20000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=150000,in,bo=20000：CUDA_VISIBLE_DEVICES=0 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=150000 data_conf.initial_data_n=20000 data_conf.boundary_data_n=20000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=40000,in,bo=25000：CUDA_VISIBLE_DEVICES=1 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=40000 data_conf.initial_data_n=25000 data_conf.boundary_data_n=25000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=80000,in,bo=25000：CUDA_VISIBLE_DEVICES=2 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=25000 data_conf.boundary_data_n=25000 hydra.job.chdir=True
- [x] step=500000,pde_data_n=150000,in,bo=25000：CUDA_VISIBLE_DEVICES=3 python train.py --config-name=Schrodinger train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=150000 data_conf.initial_data_n=25000 data_conf.boundary_data_n=25000 hydra.job.chdir=True

- step=500000,pde_data_n=80000,in,bo=10000:
- [x] layer_size=30,layer_n=2,3,4,5：CUDA_VISIBLE_DEVICES=0 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=10000 data_conf.boundary_data_n=10000 model_conf.layer.layer_n=2,3,4,5 model_conf.layer.layer_size=30 hydra.job.chdir=True
- [x] layer_size=64,layer_n=2,3,5：CUDA_VISIBLE_DEVICES=1 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=10000 data_conf.boundary_data_n=10000 model_conf.layer.layer_n=2,3,5 model_conf.layer.layer_size=64 hydra.job.chdir=True
- [x] layer_size=100,layer_n=2,3,4,5：CUDA_VISIBLE_DEVICES=2 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=10000 data_conf.boundary_data_n=10000 model_conf.layer.layer_n=2,3,4,5 model_conf.layer.layer_size=100 hydra.job.chdir=True
- [x] layer_size=200,layer_n=2,3,4,5：CUDA_VISIBLE_DEVICES=3 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=10000 data_conf.boundary_data_n=10000 model_conf.layer.layer_n=2,3,4,5 model_conf.layer.layer_size=200 hydra.job.chdir=True
- [x] layer_size=400,600,800,layer_n=2：CUDA_VISIBLE_DEVICES=6 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=10000 data_conf.boundary_data_n=10000 model_conf.layer.layer_n=2 model_conf.layer.layer_size=400,600,800 hydra.job.chdir=True

- step=500000,pde_data_n=150000,in,bo=10000:
- [x] layer_size=30,layer_n=2,3,4,5：CUDA_VISIBLE_DEVICES=0 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=150000 data_conf.initial_data_n=10000 data_conf.boundary_data_n=10000 model_conf.layer.layer_n=2,3,4,5 model_conf.layer.layer_size=30 hydra.job.chdir=True
- [x] layer_size=64,layer_n=2,3,4,5：CUDA_VISIBLE_DEVICES=1 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=150000 data_conf.initial_data_n=10000 data_conf.boundary_data_n=10000 model_conf.layer.layer_n=2,3,4,5 model_conf.layer.layer_size=64 hydra.job.chdir=True
- [x] layer_size=100,layer_n=2,3,4,5：CUDA_VISIBLE_DEVICES=4 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=150000 data_conf.initial_data_n=10000 data_conf.boundary_data_n=10000 model_conf.layer.layer_n=2,3,4,5 model_conf.layer.layer_size=100 hydra.job.chdir=True
- [x] layer_size=200,layer_n=2,3,4,5：CUDA_VISIBLE_DEVICES=5 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=150000 data_conf.initial_data_n=10000 data_conf.boundary_data_n=10000 model_conf.layer.layer_n=2,3,4,5 model_conf.layer.layer_size=200 hydra.job.chdir=True

- step=500000,pde_data_n=80000,in,bo=10000:
- [x] pde_batch_size=60000,80000,in,bo_batch_size=1000:CUDA_VISIBLE_DEVICES=1 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=10000 data_conf.boundary_data_n=10000 train_conf.main_conf.pde_batch_size=60000,80000 train_conf.main_conf.initial_batch_size=1000 train_conf.main_conf.boundary_batch_size=1000  hydra.job.chdir=True
- [x] pde_batch_size=60000,80000,in,bo_batch_size=3000:CUDA_VISIBLE_DEVICES=2 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=10000 data_conf.boundary_data_n=10000 train_conf.main_conf.pde_batch_size=60000,80000 train_conf.main_conf.initial_batch_size=3000 train_conf.main_conf.boundary_batch_size=3000  hydra.job.chdir=True
- [x] pde_batch_size=60000,80000,in,bo_batch_size=5000:CUDA_VISIBLE_DEVICES=3 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=10000 data_conf.boundary_data_n=10000 train_conf.main_conf.pde_batch_size=60000,80000 train_conf.main_conf.initial_batch_size=5000 train_conf.main_conf.boundary_batch_size=5000  hydra.job.chdir=True
- [ ] pde_batch_size=60000,80000,in,bo_batch_size=8000:CUDA_VISIBLE_DEVICES=4 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=10000 data_conf.boundary_data_n=10000 train_conf.main_conf.pde_batch_size=60000,80000 train_conf.main_conf.initial_batch_size=8000 train_conf.main_conf.boundary_batch_size=8000  hydra.job.chdir=True
- [x] pde_batch_size=80000,in,bo_batch_size=10000:CUDA_VISIBLE_DEVICES=4 python train.py --config-name=Schrodinger --multirun train_conf.main_conf.max_steps=500000 data_conf.pde_data_n=80000 data_conf.initial_data_n=10000 data_conf.boundary_data_n=10000 train_conf.main_conf.pde_batch_size=80000 train_conf.main_conf.initial_batch_size=10000 train_conf.main_conf.boundary_batch_size=10000  hydra.job.chdir=True

- learning-rate
- [x] sch=step,lr=1e-3:CUDA_VISIBLE_DEVICES=1 python train.py --config-name=Schrodinger_2 train_conf.sch=step hydra.job.chdir=True
- [x] sch=cos,lr=1e-3:CUDA_VISIBLE_DEVICES=2 python train.py --config-name=Schrodinger_2 train_conf.sch=cos hydra.job.chdir=True
- [x] sch=expon,lr=1e-3:CUDA_VISIBLE_DEVICES=3 python train.py --config-name=Schrodinger_2 train_conf.sch=expon hydra.job.chdir=True
- [x] sch=cos,lr=5e-3:CUDA_VISIBLE_DEVICES=2 python train.py --config-name=Schrodinger_2 train_conf.optim_conf.lr=5e-3 train_conf.sch=cos hydra.job.chdir=True
- [x] sch=cos,lr=5e-4:CUDA_VISIBLE_DEVICES=3 python train.py --config-name=Schrodinger_2 train_conf.optim_conf.lr=5e-4 train_conf.sch=cos hydra.job.chdir=True
- [x] sch=step,lr=5e-3:CUDA_VISIBLE_DEVICES=1 python train.py --config-name=Schrodinger_2 train_conf.optim_conf.lr=5e-3 train_conf.sch=step hydra.job.chdir=True
- [x] sch=step,lr=5e-4:CUDA_VISIBLE_DEVICES=3 python train.py --config-name=Schrodinger_2 train_conf.optim_conf.lr=5e-4 train_conf.sch=step hydra.job.chdir=True
- [x] sch=onecycle,lr=1e-3,pct_start=0.1:CUDA_VISIBLE_DEVICES=2 python train.py --config-name=Schrodinger_2 train_conf.sch=onecycle hydra.job.chdir=True
- [ ] sch=plateau:CUDA_VISIBLE_DEVICES=2 python train.py --config-name=Schrodinger_2 train_conf.sch=plateau hydra.job.chdir=True
- [ ] sch=cyclic:CUDA_VISIBLE_DEVICES=3 python train.py --config-name=Schrodinger_2 train_conf.optim=sgd train_conf.sch=cyclic hydra.job.chdir=True
- [x] sch=onecycle,lr=1e-3,pct_start=0.2:CUDA_VISIBLE_DEVICES=1 python train.py --config-name=Schrodinger_2 train_conf.sch=onecycle train_conf.sch_par.pct_start=0.2 hydra.job.chdir=True
- [x] sch=onecycle,lr=1e-3,pct_start=0.3:CUDA_VISIBLE_DEVICES=2 python train.py --config-name=Schrodinger_2 train_conf.sch=onecycle train_conf.sch_par.pct_start=0.3 hydra.job.chdir=True
- [x] sch=onecycle,lr=2e-4,pct_start=0.2:CUDA_VISIBLE_DEVICES=3 python train.py --config-name=Schrodinger_2 train_conf.sch=onecycle train_conf.sch_par.pct_start=0.2 train_conf.optim_conf.lr=2e-4 hydra.job.chdir=True

- adaptive activate_func
- [x] sch=cos,lr=1e-3,activate_func=adaptive_tanh:CUDA_VISIBLE_DEVICES=4 python train.py --config-name=Schrodinger_2 train_conf.sch=cos model_conf.layer.activate=adaptive_tanh hydra.job.chdir=True
- [x] sch=onecycle,lr=1e-3,pct_start=0.2,activate_func=adaptive_tanh:CUDA_VISIBLE_DEVICES=5 python train.py --config-name=Schrodinger_2 train_conf.sch=onecycle train_conf.sch_par.pct_start=0.2 model_conf.layer.activate=adaptive_tanh hydra.job.chdir=True
- [x] sch=cos,lr=1e-3,activate_func=adaptive_tanh,activate_scale_factor=5:CUDA_VISIBLE_DEVICES=5 python train.py --config-name=Schrodinger_2 train_conf.sch=cos model_conf.layer.activate=adaptive_tanh model_conf.layer.activate_scale_factor=5.0 hydra.job.chdir=True
- [x] sch=cos,lr=1e-3,activate_func=adaptive_tanh,activate_scale_factor=10:CUDA_VISIBLE_DEVICES=5 python train.py --config-name=Schrodinger_2 train_conf.sch=cos model_conf.layer.activate=adaptive_tanh model_conf.layer.activate_scale_factor=10.0 hydra.job.chdir=True
- [x] sch=onecycle,lr=1e-3,pct_start=0.2,activate_func=adaptive_tanh,activate_scale_factor=5:CUDA_VISIBLE_DEVICES=4 python train.py --config-name=Schrodinger_2 train_conf.sch=onecycle train_conf.sch_par.pct_start=0.2 model_conf.layer.activate=adaptive_tanh model_conf.layer.activate_scale_factor=5.0 hydra.job.chdir=True
- [x] sch=onecycle,lr=1e-3,pct_start=0.2,activate_func=adaptive_tanh,activate_scale_factor=10:CUDA_VISIBLE_DEVICES=4 python train.py --config-name=Schrodinger_2 train_conf.sch=onecycle train_conf.sch_par.pct_start=0.2 model_conf.layer.activate=adaptive_tanh model_conf.layer.activate_scale_factor=10.0 hydra.job.chdir=True
- [x] sch=onecycle,lr=1e-3,pct_start=0.2,activate_func=adaptive_tanh,max_steps=1000000:CUDA_VISIBLE_DEVICES=5 python train.py --config-name=Schrodinger_2 train_conf.sch=onecycle train_conf.sch_par.pct_start=0.2 model_conf.layer.activate=adaptive_tanh train_conf.main_conf.max_steps=1000000 hydra.job.chdir=True

- gPINNs
- [x] sch=onecycle,lr=1e-3,pct_start=0.2,activate_func=adaptive_tanh,pde_batch_size=40000,gpinn=True:CUDA_VISIBLE_DEVICES=1 python train.py --config-name=Schrodinger_2 train_conf.sch=onecycle train_conf.sch_par.pct_start=0.2 model_conf.layer.activate=adaptive_tanh model_conf.gPINNs=True train_conf.main_conf.pde_batch_size=40000 hydra.job.chdir=True
- [x] sch=onecycle,lr=1e-3,pct_start=0.2,activate_func=adaptive_tanh,pde_batch_size=80000,gpinn=True:CUDA_VISIBLE_DEVICES=2 python train.py --config-name=Schrodinger_2 train_conf.sch=onecycle train_conf.sch_par.pct_start=0.2 model_conf.layer.activate=adaptive_tanh model_conf.gPINNs=True train_conf.main_conf.pde_batch_size=80000 hydra.job.chdir=True

- SiLU
- [ ] sch=onecycle,lr=1e-3,pct_start=0.2,activate_func=silu:CUDA_VISIBLE_DEVICES=4 python train.py --config-name=Schrodinger_2 train_conf.sch=onecycle train_conf.sch_par.pct_start=0.2 model_conf.layer.activate=silu hydra.job.chdir=True
- [ ] sch=cos,lr=1e-3,activate_func=silu:CUDA_VISIBLE_DEVICES=5 python train.py --config-name=Schrodinger_2 train_conf.sch=cos model_conf.layer.activate=silu hydra.job.chdir=True


# Results

|     Model(layer_size=64,layer_n=4)      |  Loss_Total  |  Score  |               model_path               |
|:---------------------------------------:|:------------:|:-------:|:--------------------------------------:|
|           baseline-step=20000           | 2.63425e-03  | 0.18794 |                                        |
|               step=100000               | 8.09942e-05  | 0.03687 | 2023-10-24/23-18-43/Schrodinger_99500  |
|               step=500000               | 1.43624e-05  | 0.01810 | 2023-10-25/00-22-13/Schrodinger_483500 |
|              step=1000000               | 9.23105e-06  | 0.01728 | 2023-10-25/08-35-17/Schrodinger_907000 | 
|             pde_data_n=2000             | 3.19287e-05  | 0.13993 | 2023-10-25/21-33-49/Schrodinger_94000  | 
|             pde_data_n=4000             | 8.63019e-05  | 0.06115 | 2023-10-25/22-30-17/Schrodinger_93000  | 
|            pde_data_n=10000             | 9.86333e-05  | 0.04009 | 2023-10-25/23-40-54/Schrodinger_94500  | 
|            pde_data_n=20000             | 9.83613e-05  | 0.03539 | 2023-10-26/00-42-47/Schrodinger_90000  | 
|            pde_data_n=40000             | 7.73657e-05  | 0.03302 | 2023-10-26/01-46-17/Schrodinger_98000  | 
|            pde_data_n=60000             | 8.09942e-05  | 0.03687 | 2023-10-24/23-18-43/Schrodinger_99500  | 
|            pde_data_n=80000             | 8.80178e-05  | 0.03223 | 2023-10-26/14-39-14/Schrodinger_95500  | 
|            pde_data_n=100000            | 7.74835e-05  | 0.03303 | 2023-10-26/15-44-56/Schrodinger_88000  | 
|            pde_data_n=150000            | 8.69922e-05  | 0.03628 | 2023-10-26/20-02-15/Schrodinger_98500  | 
|       pde_data_n=80000/in,bo=500        | 9.27920e-05  | 0.0374  | 2023-10-26/21-14-00/Schrodinger_98000  | 
|       pde_data_n=80000/in,bo=1000       | 7.68144e-05  | 0.02921 | 2023-10-26/22-53-00/Schrodinger_92000  |
|       pde_data_n=80000/in,bo=3000       | 8.43157e-05  | 0.03037 | 2023-10-27/00-36-51/Schrodinger_98000  |
|       pde_data_n=80000/in,bo=5000       | 8.87872e-05  | 0.02812 | 2023-10-27/08-43-21/Schrodinger_97000  |
|       pde_data_n=40000/in,bo=500        | 9.40486e-05  |         | 2023-10-27/10-49-50/Schrodinger_95000  |
|       pde_data_n=40000/in,bo=1000       | 8.02257e-05  | 0.02897 | 2023-10-27/15-00-27/Schrodinger_97000  |
|       pde_data_n=40000/in,bo=3000       | 8.13654e-05  | 0.02781 | 2023-10-27/16-09-54/Schrodinger_95000  |
| step=500000,pde_data_n=40000,in,bo=3000 | 1.43172e-05  | 0.01723 | 2023-10-27/17-14-50/Schrodinger_440500 |
| step=500000,pde_data_n=80000,in,bo=1000 | 1.34394e-05  | 0.01695 | 2023-10-28/09-13-06/Schrodinger_495000 |

| Model(layer_size=64,layer_n=4,step=500000) | Loss_Total  |  Score  |               model_path               |
|:------------------------------------------:|:-----------:|:-------:|:--------------------------------------:|
|             pde_data_n=40000,              | 1.40046e-05 | 0.0194  | 2023-11-10/12-11-47/Schrodinger_467000 |
|              pde_data_n=80000              | 1.42688e-05 | 0.02008 | 2023-11-10/12-13-51/Schrodinger_496500 |
|             pde_data_n=100000              | 1.47698e-05 | 0.0178  | 2023-11-10/12-14-31/Schrodinger_494500 |
|             pde_data_n=150000              | 1.30376e-05 | 0.01757 | 2023-11-10/12-15-14/Schrodinger_491000 |
|             pde_data_n=200000              | 1.52866e-05 | 0.01815 | 2023-11-10/18-31-08/Schrodinger_481500 |
|             pde_data_n=250000              | 1.42191e-05 | 0.01945 | 2023-11-10/18-31-22/Schrodinger_488500 |

| Model(layer_size=64,layer_n=4,step=500000) | Loss_Total  |  Score  |               model_path               |
|:------------------------------------------:|:-----------:|:-------:|:--------------------------------------:|
|         pde_data_n=40000,in,bo=500         | 1.51491e-05 | 0.01941 | 2023-11-13/14-33-16/Schrodinger_440000 |
|        pde_data_n=40000,in,bo=1000         | 1.44112e-05 | 0.01867 | 2023-11-13/14-34-26/Schrodinger_482000 |
|        pde_data_n=40000,in,bo=3000         | 1.43172e-05 | 0.01723 | 2023-10-27/17-14-50/Schrodinger_440500 |
|        pde_data_n=40000,in,bo=5000         | 1.59779e-05 | 0.01742 | 2023-11-13/14-35-26/Schrodinger_473500 |
|        pde_data_n=40000,in,bo=8000         | 1.22387e-05 | 0.01717 | 2023-11-13/14-50-32/Schrodinger_469000 |
|        pde_data_n=40000,in,bo=10000        | 1.38218e-05 | 0.01682 | 2023-11-13/14-51-06/Schrodinger_490500 |
|        pde_data_n=40000,in,bo=15000        | 1.34566e-05 | 0.01714 | 2023-11-14/08-52-06/Schrodinger_493500 |
|        pde_data_n=40000,in,bo=20000        | 1.45672e-05 | 0.01689 | 2023-11-14/08-53-23/Schrodinger_431000 |
|        pde_data_n=40000,in,bo=25000        | 1.40589e-05 | 0.0171  | 2023-11-14/08-54-09/Schrodinger_480500 |
|         pde_data_n=80000,in,bo=500         | 1.45042e-05 | 0.01794 | 2023-11-13/14-36-22/Schrodinger_494500 |
|        pde_data_n=80000,in,bo=1000         | 1.34394e-05 | 0.01695 | 2023-10-28/09-13-06/Schrodinger_495000 |
|        pde_data_n=80000,in,bo=3000         | 1.55204e-05 | 0.01753 | 2023-11-13/14-38-27/Schrodinger_472500 |
|        pde_data_n=80000,in,bo=5000         | 1.34063e-05 | 0.01723 | 2023-11-13/14-39-18/Schrodinger_442000 |
|        pde_data_n=80000,in,bo=8000         | 1.42084e-05 | 0.0168  | 2023-11-13/14-51-54/Schrodinger_499500 |
|        pde_data_n=80000,in,bo=10000        | 1.34416e-05 | 0.01641 | 2023-11-13/14-52-41/Schrodinger_497500 |
|        pde_data_n=80000,in,bo=15000        | 1.43389e-05 | 0.01761 | 2023-11-14/08-52-41/Schrodinger_480500 |
|        pde_data_n=80000,in,bo=20000        | 1.40472e-05 | 0.01714 | 2023-11-14/08-53-35/Schrodinger_468000 |
|        pde_data_n=80000,in,bo=25000        | 1.45041e-05 |         | 2023-11-14/08-54-25/Schrodinger_483000 |
|        pde_data_n=150000,in,bo=500         | 1.42564e-05 |         | 2023-11-13/14-40-05/Schrodinger_466000 |
|        pde_data_n=150000,in,bo=1000        | 1.27594e-05 |         | 2023-11-13/14-41-11/Schrodinger_498000 |
|        pde_data_n=150000,in,bo=3000        | 1.52765e-05 |         | 2023-11-13/14-41-45/Schrodinger_493000 |
|        pde_data_n=150000,in,bo=5000        | 1.32526e-05 | 0.01688 | 2023-11-13/14-42-20/Schrodinger_465500 |
|        pde_data_n=150000,in,bo=8000        | 1.67202e-05 |         | 2023-11-13/14-53-13/Schrodinger_411500 |
|       pde_data_n=150000,in,bo=10000        | 1.60290e-05 | 0.01682 | 2023-11-13/14-53-57/Schrodinger_424000 |
|       pde_data_n=150000,in,bo=15000        | 1.37342e-05 | 0.01791 | 2023-11-14/08-53-04/Schrodinger_497500 |
|       pde_data_n=150000,in,bo=20000        | 1.60017e-05 |         | 2023-11-14/08-53-37/Schrodinger_479000 |
|       pde_data_n=150000,in,bo=25000        | 1.36148e-05 |         | 2023-11-14/08-54-37/Schrodinger_470500 |

| Model(pde_data_n=80000,in,bo=10000,step=500000) | Loss_Total  |  Score  |                model_path                |
|:-----------------------------------------------:|:-----------:|:-------:|:----------------------------------------:|
|             layer_size=30,layer_n=2             | 2.17594e-04 |         | 2023-11-15/08-51-02/0/Schrodinger_485500 |
|             layer_size=30,layer_n=3             | 4.03427e-05 |         | 2023-11-15/08-51-02/1/Schrodinger_493000 |
|             layer_size=30,layer_n=4             | 2.73124e-05 |         | 2023-11-15/08-51-02/2/Schrodinger_495500 |
|             layer_size=30,layer_n=5             | 2.25104e-05 |         | 2023-11-15/08-51-02/3/Schrodinger_480500 |
|             layer_size=64,layer_n=2             | 5.61535e-05 |         | 2023-11-15/08-53-13/0/Schrodinger_480500 |
|             layer_size=64,layer_n=3             | 1.55498e-05 | 0.01685 | 2023-11-15/08-53-13/1/Schrodinger_487500 |
|             layer_size=64,layer_n=4             | 1.34416e-05 | 0.01641 |  2023-11-13/14-52-41/Schrodinger_497500  |
|             layer_size=64,layer_n=5             | 1.37757e-05 | 0.01972 | 2023-11-15/08-53-13/2/Schrodinger_445500 |
|            layer_size=100,layer_n=2             | 3.46111e-05 |         | 2023-11-15/08-53-28/0/Schrodinger_499000 |
|            layer_size=100,layer_n=3             | 1.55129e-05 |         | 2023-11-15/08-53-28/1/Schrodinger_497500 |
|            layer_size=100,layer_n=4             | 1.27152e-05 | 0.01739 | 2023-11-15/08-53-28/2/Schrodinger_497500 |
|            layer_size=100,layer_n=5             | 1.85961e-05 |         | 2023-11-15/08-53-28/3/Schrodinger_491500 |
|            layer_size=200,layer_n=2             | 3.25953e-05 |         | 2023-11-15/08-53-49/0/Schrodinger_462000 |
|            layer_size=200,layer_n=3             | 1.99498e-05 |         | 2023-11-15/08-53-49/1/Schrodinger_439000 |
|            layer_size=200,layer_n=4             | 1.50480e-05 |         | 2023-11-15/08-53-49/2/Schrodinger_488500 |
|            layer_size=200,layer_n=5             | 1.12781e-04 |         | 2023-11-15/08-53-49/3/Schrodinger_495500 |
|            layer_size=400,layer_n=2             | 4.70356e-05 |         | 2023-11-16/14-49-53/0/Schrodinger_494000 |

| Model(pde_data_n=80000,in,bo=10000,layer_size=64,layer_n=4) | Loss_Total  |  Score  |                model_path                |
|:-----------------------------------------------------------:|:-----------:|:-------:|:----------------------------------------:|
|         pde_batch_size=40000,in,bo_batch_size=1000          | 1.22674e-05 |         | 2023-11-17/09-47-34/0/Schrodinger_490500 |
|         pde_batch_size=60000,in,bo_batch_size=1000          | 1.21208e-05 |         | 2023-11-20/14-06-46/0/Schrodinger_460500 |
|         pde_batch_size=80000,in,bo_batch_size=1000          | 1.21117e-05 | 0.01649 | 2023-11-20/14-06-46/1/Schrodinger_475500 |
|         pde_batch_size=40000,in,bo_batch_size=3000          | 1.28156e-05 |         | 2023-11-17/09-49-47/0/Schrodinger_495500 |
|         pde_batch_size=60000,in,bo_batch_size=3000          | 1.15495e-05 | 0.01655 | 2023-11-20/14-09-19/0/Schrodinger_486500 |
|         pde_batch_size=80000,in,bo_batch_size=3000          | 1.16337e-05 | 0.01583 | 2023-11-20/14-09-19/1/Schrodinger_496000 |
|         pde_batch_size=40000,in,bo_batch_size=5000          | 1.11789e-05 | 0.01599 | 2023-11-17/09-54-24/0/Schrodinger_497500 |
|         pde_batch_size=60000,in,bo_batch_size=5000          | 1.23536e-05 |         | 2023-11-20/14-14-28/0/Schrodinger_470500 |
|         pde_batch_size=80000,in,bo_batch_size=5000          | 1.19882e-05 | 0.01663 | 2023-11-20/14-14-28/1/Schrodinger_470000 |
|         pde_batch_size=40000,in,bo_batch_size=8000          | 1.22906e-05 | 0.01688 | 2023-11-17/09-54-42/0/Schrodinger_491000 |
|         pde_batch_size=60000,in,bo_batch_size=8000          | 1.16038e-05 | 0.01638 | 2023-11-21/09-02-52/0/Schrodinger_484500 |
|         pde_batch_size=80000,in,bo_batch_size=8000          | 1.09211e-05 | 0.01588 | 2023-11-21/09-02-52/1/Schrodinger_486000 |
|         pde_batch_size=40000,in,bo_batch_size=10000         | 1.26466e-05 | 0.01673 | 2023-11-17/09-54-51/0/Schrodinger_497500 |
|         pde_batch_size=60000,in,bo_batch_size=10000         | 1.23257e-05 | 0.01628 | 2023-11-17/09-54-51/1/Schrodinger_486000 |
|         pde_batch_size=80000,in,bo_batch_size=10000         | 1.22852e-05 | 0.01645 | 2023-11-20/14-18-54/0/Schrodinger_478000 |

pde_data_n=80000,in,bo=10000,layer_size=64,layer_n=4,pde_batch_size=80000,in,bo_batch_size=3000

|                                          Model                                          | Loss_Total  |  Score  |               model_path                |
|:---------------------------------------------------------------------------------------:|:-----------:|:-------:|:---------------------------------------:|
|                                    sch=step,lr=1e-3                                     | 2.22446e-06 | 0.01272 | 2023-11-21/21-31-35/Schrodinger_497000  |
|                                     sch=cos,lr=1e-3                                     | 2.36820e-06 | 0.0127  | 2023-11-21/21-31-48/Schrodinger_498500  |
|                           sch=onecycle,lr=1e-3,pct_start=0.1                            | 1.79403e-06 | 0.01271 | 2023-11-23/14-03-51/Schrodinger_497000  |
|                                     sch=cos,lr=5e-3                                     | 3.02756e-06 | 0.01282 | 2023-11-22/21-04-05/Schrodinger_498500  |
|                                     sch=cos,lr=5e-4                                     | 4.10847e-06 | 0.01288 | 2023-11-22/21-04-16/Schrodinger_491500  |
|                                    sch=step,lr=5e-3                                     | 2.11799e-06 | 0.01271 | 2023-11-23/08-26-12/Schrodinger_497000  |
|                                    sch=step,lr=5e-4                                     | 5.72323e-06 |         | 2023-11-23/14-08-22/Schrodinger_491500  |
|                           sch=onecycle,lr=1e-3,pct_start=0.2                            | 2.04241e-06 | 0.01271 | 2023-11-27/08-44-54/Schrodinger_497000  |
|                           sch=onecycle,lr=1e-3,pct_start=0.3                            | 2.07700e-06 | 0.01272 | 2023-11-27/08-45-02/Schrodinger_497000  |
|                           sch=onecycle,lr=2e-4,pct_start=0.2                            | 1.67650e-05 |         | 2023-11-27/08-45-24/Schrodinger_492500  |
|                       sch=cos,lr=1e-3,activate_func=adaptive_tanh                       | 1.56126e-06 | 0.01273 | 2023-11-27/09-21-07/Schrodinger_496000  |
|             sch=onecycle,lr=1e-3,pct_start=0.2,activate_func=adaptive_tanh              | 1.57909e-06 | 0.0127  | 2023-11-28/22-58-21/Schrodinger_497000  |
|           sch=cos,lr=1e-3,activate_func=adaptive_tanh,activate_scale_factor=5           | 1.58588e-06 | 0.01277 | 2023-11-28/22-58-44/Schrodinger_498500  |
|          sch=cos,lr=1e-3,activate_func=adaptive_tanh,activate_scale_factor=10           | 1.61653e-06 | 0.01279 | 2023-11-28/22-59-00/Schrodinger_497500  |
| sch=onecycle,lr=1e-3,pct_start=0.2,activate_func=adaptive_tanh,activate_scale_factor=5  | 1.21099e-06 | 0.01273 | 2023-11-29/01-28-40/Schrodinger_497000  |
| sch=onecycle,lr=1e-3,pct_start=0.2,activate_func=adaptive_tanh,activate_scale_factor=10 | 1.28245e-06 | 0.01273 | 2023-11-29/01-29-47/Schrodinger_497000  |

pde_data_n=80000,in,bo=10000,layer_size=64,layer_n=4,in,bo_batch_size=3000

|                                    Model(gPINNs)                                    | Loss_Total  |  Score  |               model_path               |
|:-----------------------------------------------------------------------------------:|:-----------:|:-------:|:--------------------------------------:|
| sch=onecycle,lr=1e-3,pct_start=0.2,activate_func=adaptive_tanh,pde_batch_size=40000 | 2.22601e-06 | 0.01277 | 2023-11-30/12-06-30/Schrodinger_497500 |
| sch=onecycle,lr=1e-3,pct_start=0.2,activate_func=adaptive_tanh,pde_batch_size=80000 | 2.00228e-06 | 0.01275 | 2023-11-30/12-07-20/Schrodinger_497000 |
