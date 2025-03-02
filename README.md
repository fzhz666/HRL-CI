# HRL-CI

# Overview

This is a navigation task for  mapless moving target navigation with communication interruption. Our HRL-CI method can greatly improve the success rate and navigation efficiency compared with the pH-DRL method in the case of communication interruption. This project will open source our code, not only to provide developers with a navigation task platform, but also to make the code public.



## Dependencies

First, go to this link and follow the tutorial to build the robot platform environment. 
https://blog.csdn.net/m0_45131654/article/details/121256885

The basic robot platform is built based on the reference paper.
https://ieeexplore.ieee.org/document/9340948?denied=
 [Paper](https://arxiv.org/abs/2003.01157) 

```
conda create -n hrl_ci python=3.9.12 pip=23.0.1
conda activate hrl_ci
pip install -r requirements.txt
```
