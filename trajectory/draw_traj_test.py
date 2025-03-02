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
from collections import deque


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


def fusion_traj(pred_target_traj):
    a = 0.5
    shuzi = 3
    # 在训练循环开始之前初始化一个队列，用于存储预测目标轨迹
    deque_number = 3  # 设置有几个队列
    # target_traj_queue = deque(maxlen=deque_number)  # 最多存储5条轨迹
    # 只需要再多添加这一句就好（对于验证代码来讲）
    target_traj_queue = pred_target_traj
    pred_target_traj = target_traj_queue[-1]
    ###
    while shuzi != len(target_traj_queue):
        target_traj_queue.append(pred_target_traj)  # 目标预测路径队列初始化，全部设置为第一条
        # shuzi += 1

    # 初始化综合的预测路径
    combined_pred_traj = np.zeros_like(pred_target_traj)
    combined_pred_traj += (a ** (deque_number - 1)) * target_traj_queue[0]  # 第一条预测路径乘以a
    weight = 1 - a
    # print('len(target_traj_queue) = ', len(target_traj_queue))
    for i in range(1, len(target_traj_queue)):
        combined_pred_traj += weight * target_traj_queue[len(target_traj_queue) - i]  # 后续路径乘以(1-a)的累计系数
        weight *= a  # 更新权重，叠加(1-a)的次方
        # 手动调整综合轨迹的起点为目标点的起点
    offset = pred_target_traj[0] - combined_pred_traj[0]
    combined_pred_traj += offset
    pred_target_traj = combined_pred_traj
    target_traj_queue[-1] = pred_target_traj
    return target_traj_queue


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
traj = tg.gen_random_rrt_track(-8, 8, 1050, ((-10, 10), (-10, 10)), poly_list)
traj = np.array(traj)

rand_paths_robot_list = pickle.load(open('eval_rand_paths_robot_pose.p', 'rb'))
robot_init_list = rand_paths_robot_list[0][:]
target_paths_list = rand_paths_robot_list[1][:]

# traj = np.array(rand_paths_robot_list[0])
# target_init_pos = target_paths_list[0]


# predictor initialization
# long 30-150, short 30-20
tau = 30
pred_len = 150
kf_predictor = KFPredictor(tau, pred_len)

images = []
ita = 0
all_pred_traj = []  # 用于保存所有的 pred_traj
pred_images = []
pred_fusion_images = []
all_fusion_pred_traj = []
fusion_traj_queue = deque(maxlen=3)
show_numb = 100
step = 15
disappear = 40

print("traj.__len__() = ", traj.__len__())
while ita < traj.__len__() - pred_len:
    # predict trajectory
    quotient = ita // 100
    remainder = ita % 100
    flag = 0
    if disappear <= remainder <= 100:
        flag = 1
    else:
        flag = 0

    if flag == 0:
        # 这里还要加入通讯中断的逻辑才算是正确的
        observe_start = max(0, ita - tau + 1)
        traj_observed = traj[observe_start: ita + 1]
        pred_traj = kf_predictor.predict(traj_observed, ita)
    elif flag == 1:
        traj_observed = traj[quotient * 100 + disappear - tau: quotient * 100 + disappear]
        pred_traj = kf_predictor.predict(traj_observed, ita)


    pred_start = ita + 1

    # 这样是错误的，没有把历史的预测轨迹读进去
    fusion_traj_queue.append(pred_traj)
    fusion_traj_queue = fusion_traj(fusion_traj_queue)
    all_fusion_pred_traj.append(fusion_traj_queue[-1])

    all_pred_traj.append(pred_traj)  # 保存当前的 pred_traj
    # compute MSE Loss
    if ita <= len(traj) - pred_len - tau:
        MSELoss = np.square(np.subtract(traj[pred_start: pred_start + pred_len], pred_traj)).mean()
        # print('MSE Loss: ', MSELoss)
    # plot
    image1, = ax.plot(traj[: pred_start, 0], traj[: pred_start, 1], color='skyblue', linewidth=1, marker='o',
                      markersize=1.6)
    image2, = ax.plot(traj_observed[:, 0], traj_observed[:, 1], color='royalblue',
                      linewidth=1, marker='o', markersize=1.6)
    image3, = ax.plot(traj[pred_start: pred_start + pred_len, 0], traj[pred_start: pred_start + pred_len, 1],
                      color='limegreen', linewidth=1, marker='o', markersize=1.6)

    if ita % step == 0:
        img_fusion, = ax.plot(all_fusion_pred_traj[-1][:show_numb, 0], all_fusion_pred_traj[-1][:show_numb, 1], color='green', linewidth=0.5, marker='o',
                       markersize=0.8)
        pred_fusion_images.append(img_fusion)
    # image4, = ax.plot(pred_traj[:, 0], pred_traj[:, 1], color='yellow', linewidth=0.5, marker='o', markersize=0.8)

    # for past_pred_traj in all_pred_traj:
    if ita % step == 0:
        img, = ax.plot(all_pred_traj[-1][:show_numb, 0], all_pred_traj[-1][:show_numb, 1], color='red', linewidth=0.5, marker='o',
                       markersize=0.8)
        pred_images.append(img)

    # plot target
    target = patches.RegularPolygon((traj[pred_start, 0].item(), traj[pred_start, 1].item()), 4, radius=0.2 * np.sqrt(2),
                                    color='tomato')
    target_img = ax.add_patch(target)
    title = ax.text(0.5, 1.05, "MSE = {:.4f}".format(MSELoss),
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes, )
    # images.append([image1, image2, image3, image4, target_img, title])

    frame_images = [image1, image2, image3, target_img, title] + pred_images + pred_fusion_images
    images.append(frame_images)

    ita += 1

print("ita = ", ita)
print("images.shape = ", len(images))
ani = animation.ArtistAnimation(fig, images, interval=50, blit=False)  #50
plt.show()


# 定义播放次数
num_plays = 3  # 这里设置为播放 3 次
current_play = 0
frame_index = 0


# def update(frame):
#     global current_play, frame_index
#     for img in ax.lines:
#         img.remove()
#     for img in images[frame_index]:
#         ax.add_artist(img)
#     frame_index += 1
#     if frame_index >= len(images):
#         frame_index = 0
#         current_play += 1
#     if current_play >= num_plays:
#         ani.event_source.stop()  # 停止动画
#     return ax.lines
#
#
# ani = animation.FuncAnimation(fig, update, frames=len(images), interval=10, blit=False)
# plt.show()


