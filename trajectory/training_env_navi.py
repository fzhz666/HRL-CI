import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import trajectory_generation as tg
from trajectory_prediction import KFPredictor
import matplotlib.patches as patches
import sys
sys.path.append('../../')


def encode_trajectory(traj_predict, robot_pose):
    '''
    Compuate relative distance and direction between robot and
    predicted trajectory of motion target
    '''
    # relative distance
    trajectory = np.array(traj_predict)
    delta_x = trajectory[:, 0] - robot_pose[0]
    delta_y = trajectory[:, 1] - robot_pose[1]
    dist = np.sqrt(delta_x ** 2 + delta_y ** 2)
    # relative direction
    direction = np.arctan2(delta_y, delta_x)
    encoded_trajectory = np.column_stack((direction, dist))
    return encoded_trajectory


# plot obstacles
from training.utils import *
fig, ax = plt.subplots(figsize=(10, 10))

# Define 4 training environments
env1_range, env1_poly_list, env1_raw_poly_list, env1_goal_list, env1_init_list = gen_rand_list_env1(100)
env2_range, env2_poly_list, env2_raw_poly_list, env2_goal_list, env2_init_list = gen_rand_list_env2(200)
env3_range, env3_poly_list, env3_raw_poly_list, env3_goal_list, env3_init_list = gen_rand_list_env3(300)
env4_range, env4_poly_list, env4_raw_poly_list, env4_goal_list, env4_init_list = gen_rand_list_env4(400)
env_raw_poly_list = [env1_raw_poly_list, env2_raw_poly_list, env3_raw_poly_list, env4_raw_poly_list]
for i in range(4):
    for obs in env_raw_poly_list[i]:
        x = [obs[num][0] for num in range(len(obs))]
        y = [obs[num][1] for num in range(len(obs))]
        x.append(obs[0][0])
        y.append(obs[0][1])
        ax.plot(x, y, 'k-')
ax.set_xticks(np.linspace(-10, 10, 21))
ax.set_yticks(np.linspace(-10, 10, 21))

# generate trajectory for test
robot_position = env2_init_list[20][:2]
target_init_pos = env2_goal_list[20]
robot_pos_list = [robot_position]
traj = tg.gen_random_rrt_track(target_init_pos[0], target_init_pos[1], 650, env2_range, env2_poly_list)
traj = np.array(traj)

# predictor initialization
tau = 50
pred_len = 100
kf_predictor = KFPredictor(tau, pred_len)

images = []
ita = 0
while ita < traj.__len__() - pred_len:
    # predict trajectory
    observe_start = max(0, ita - tau + 1)
    traj_observed = traj[observe_start: ita + 1]
    pred_traj = kf_predictor.predict(traj_observed, ita)
    pred_start = ita + 1
    # compute MSE Loss
    if ita <= len(traj) - pred_len - tau:
        MSELoss = np.square(np.subtract(traj[pred_start: pred_start + pred_len], pred_traj)).mean()
        print('MSE Loss: ', MSELoss)
    # choose a navigation goal and make a move for robot
    encoded_target_traj = encode_trajectory(pred_traj, robot_position)
    goal_option = np.argmin(encoded_target_traj, axis=0)[1]
    goal = encoded_target_traj[goal_option]
    x_next = robot_position[0] + 0.03 * np.cos(goal[0])
    y_next = robot_position[1] + 0.03 * np.sin(goal[0])
    robot_pos_list.append([x_next, y_next])
    robot_position = [x_next, y_next]
    # plot
    image1, = ax.plot(traj[: pred_start, 0], traj[: pred_start, 1], color='skyblue', linewidth=1, marker='o',
                      markersize=1.6)
    image2, = ax.plot(traj[pred_start - tau: pred_start, 0], traj[pred_start - tau: pred_start, 1], color='royalblue',
                      linewidth=1, marker='o', markersize=1.6)
    image3, = ax.plot(traj[pred_start: pred_start + pred_len, 0], traj[pred_start: pred_start + pred_len, 1],
                      color='limegreen', linewidth=1, marker='o', markersize=1.6)
    image4, = ax.plot(pred_traj[:, 0], pred_traj[:, 1], color='red', linewidth=0.5, marker='o', markersize=0.8)
    # plot target
    target = patches.RegularPolygon((traj[pred_start, 0], traj[pred_start, 1]), 4, 0.2 * np.sqrt(2),
                                    color='tomato')
    target_img = ax.add_patch(target)
    # plot selected goal
    option = patches.Circle((pred_traj[goal_option, 0], pred_traj[goal_option, 1]), radius=0.15, color='orange')
    option_img = ax.add_patch(option)
    # plot robot
    robot = patches.Circle((robot_position[0], robot_position[1]), radius=0.2,
                                    color='royalblue')
    robot_img = ax.add_patch(robot)
    image5, = ax.plot([p[0] for p in robot_pos_list], [p[1] for p in robot_pos_list], color='blue', linewidth=0.5, marker='o', markersize=0.8)
    # plot distance
    relative_dis = np.array([robot_position, pred_traj[goal_option]])
    image6, = ax.plot(relative_dis[:, 0], relative_dis[:, 1], color='orange', linewidth=0.5, marker='o', markersize=0.8)
    title = ax.text(0.5, 1.05, "MSE = {:.4f}".format(MSELoss),
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes, )
    images.append([image1, image2, image3, image4, target_img, option_img, robot_img, image5, image6, title])

    ita += 1

ani = animation.ArtistAnimation(fig, images, interval=50, blit=False)  #50
plt.show()