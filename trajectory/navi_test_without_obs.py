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
import geopandas as gpd
from evaluation.eval_simulation import utils
fig, ax = plt.subplots(figsize=(10, 10))

poly_list, _ = utils.gen_test_env_poly_list_env()
for i, obs in enumerate(poly_list):
    if i > 0:
        p = gpd.GeoSeries(obs)
        p.plot(color='k', ax=ax)
    else:
        ax.plot(*obs.xy, 'k-')
ax.set_xticks(np.linspace(-10, 10, 21))
ax.set_yticks(np.linspace(-10, 10, 21))

# generate trajectory for test
start_goal_pos = pickle.load(open("../evaluation/eval_simulation/eval_positions.p", "rb"))
robot_position = start_goal_pos[0][2][:2]
target_init_pos = start_goal_pos[1][2]
robot_pos_list = [robot_position]
# try some noise on trajectory
# traj = tg.gen_random_rrt_track(target_init_pos[0], target_init_pos[1], 1005, ((-10, 10), (-10, 10)), poly_list)
traj = tg.gen_random_rrt_track(-8, 8, 1050, ((-10, 10), (-10, 10)), poly_list)
traj = np.array(traj)
# traj = traj + np.random.normal(0, 0.015, (1005, 2))

# predictor initialization
# long 30-150, short 30-20
tau = 30
pred_len = 150
kf_predictor = KFPredictor(tau, pred_len)

images = []
ita = 0

print("traj.__len__() = ", traj.__len__())
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
    # x_next = robot_position[0] + 0.04 * np.cos(goal[0])
    # y_next = robot_position[1] + 0.04 * np.sin(goal[0])
    # robot_pos_list.append([x_next, y_next])
    # robot_position = [x_next, y_next]
    # plot
    image1, = ax.plot(traj[: pred_start, 0], traj[: pred_start, 1], color='skyblue', linewidth=1, marker='o',
                      markersize=1.6)
    image2, = ax.plot(traj_observed[:, 0], traj_observed[:, 1], color='royalblue',
                      linewidth=1, marker='o', markersize=1.6)
    # image2, = ax.plot(filtered_traj[:, 0], filtered_traj[:, 1], color='royalblue',
    #                   linewidth=1, marker='o', markersize=1.6)
    image3, = ax.plot(traj[pred_start: pred_start + pred_len, 0], traj[pred_start: pred_start + pred_len, 1],
                      color='limegreen', linewidth=1, marker='o', markersize=1.6)
    image4, = ax.plot(pred_traj[:, 0], pred_traj[:, 1], color='red', linewidth=0.5, marker='o', markersize=0.8)
    # plot target
    target = patches.RegularPolygon((traj[pred_start, 0], traj[pred_start, 1]), 4, 0.2 * np.sqrt(2),
                                    color='tomato')
    # target = patches.RegularPolygon((filtered_traj[-1, 0], filtered_traj[-1, 1]), 4, 0.2 * np.sqrt(2),
    #                                 color='tomato')
    target_img = ax.add_patch(target)
    # plot selected goal
    option = patches.Circle((pred_traj[goal_option, 0], pred_traj[goal_option, 1]), radius=0.15, color='orange')
    option_img = ax.add_patch(option)
    # plot robot
    # robot = patches.Circle((robot_position[0], robot_position[1]), radius=0.2,
    #                                 color='royalblue')
    # robot_img = ax.add_patch(robot)
    # image5, = ax.plot([p[0] for p in robot_pos_list], [p[1] for p in robot_pos_list], color='blue', linewidth=0.5, marker='o', markersize=0.8)
    # plot distance
    # relative_dis = np.array([robot_position, pred_traj[goal_option]])
    # image6, = ax.plot(relative_dis[:, 0], relative_dis[:, 1], color='orange', linewidth=0.5, marker='o', markersize=0.8)
    title = ax.text(0.5, 1.05, "MSE = {:.4f}".format(MSELoss),
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes, )
    # images.append([image1, image2, image3, image4, target_img, option_img, robot_img, image5, image6, title])
    images.append([image1, image2, image3, image4, target_img, option_img, title])

    ita += 1

ani = animation.ArtistAnimation(fig, images, interval=50, blit=False)  #50
plt.show()