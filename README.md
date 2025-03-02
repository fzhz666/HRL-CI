# HRL-CI

# Overview

This is a navigation task for  mapless moving target navigation with communication interruption. Our HRL-CI method can greatly improve the success rate and navigation efficiency compared with the pH-DRL method in the case of communication interruption. This project will open source our code, not only to provide developers with a navigation task platform, but also to make the code public.



## Dependencies

First, go to this link and follow the tutorial to build the robot platform environment. 
https://blog.csdn.net/m0_45131654/article/details/121256885

The basic robot platform is built based on the reference  [Paper](https://arxiv.org/abs/2003.01157) .

```
conda create -n hrl_ci python=3.8.18 pip=23.2.1
conda activate hrl_ci
pip install -r requirements.txt
```

## Inference

```bash
roslaunch turtlebot_lidar turtlebot_world.launch 
```

```bash
rosrun simple_laserscan laserscan_simple
```

```bash
conda activate hrl_ci
cd /evaluation/eval_simulation
python run_xxx_eval.py --save 0 --cuda 1 
```



## Train

### Prepare datasets
* Download the datasets that present in `/configs/datasets.yaml` and modify the corresponding paths.
* You could prepare you own datasets according to the formates of files in `./datasets`.
* If you use UVO dataset, you need to process the json following `./datasets/Preprocess/uvo_process.py`
* You could refer to `run_dataset_debug.py` to verify you data is correct.

### Prepare initial weight
* If your would like to train from scratch, convert the downloaded SD weights to control copy by running:
```bash
sh ./scripts/convert_weight.sh  
```
### Start training
* Modify the training hyper-parameters in `run_train_anydoor.py` Line 26-34 according to your training resources. We verify that using 2-A100 GPUs with batch accumulation=1 could get satisfactory results after 300,000 iterations.


* Start training by executing: 
```bash
sh ./scripts/train.sh  
```

