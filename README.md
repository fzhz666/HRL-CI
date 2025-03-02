# HRL-CI

# Overview

This is a navigation task for  mapless moving target navigation with communication interruption. Our HRL-CI method can greatly improve the success rate and navigation efficiency compared with the [pH-DRL](https://www.sciencedirect.com/science/article/abs/pii/S089360802300309X) method in the case of communication interruption. This project will open source our code, not only to provide developers with a navigation task platform, but also to make the code public.



## Dependencies

First, go to this link and follow the tutorial to build the robot platform environment. 
https://blog.csdn.net/m0_45131654/article/details/121256885

The basic robot platform is built based on the reference  [Paper](https://arxiv.org/abs/2003.01157) .

```
conda create -n hrl_ci python=3.8.18 pip=23.2.1
conda activate hrl_ci
pip install -r requirements.txt
```

## Code Availability

More source code for HRL-CI is being prepared. We will release the code as soon as the relevant paper is published. Please stay tuned for updates!


## Inference
You need to open three command terminals and execute the following commands in order

```bash
roslaunch turtlebot_lidar turtlebot_world.launch 
```

```bash
rosrun simple_laserscan laserscan_simple
```

```bash
conda activate hrl_ci
cd /evaluation/eval_simulation
python run_inf_eval.py --save 0 --cuda 1 
```



## Train
You need to open three command terminals and execute the following commands in order

```bash
roslaunch turtlebot_lidar turtlebot_world_train.launch 
```

```bash
rosrun simple_laserscan laserscan_simple
```

```bash
conda activate hrl_ci
cd /training/train_phddpg
python train_meta_disappear_step_change.py

