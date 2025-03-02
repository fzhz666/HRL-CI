import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import trajectory_generation as tg
from trajectory_prediction import KFPredictor
import matplotlib.patches as patches
import sys
sys.path.append('../../')
from evaluation.eval_simulation import utils

# plot obstacles
import geopandas as gpd
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
target_init_pos = start_goal_pos[1][2]
# traj = tg.gen_random_rrt_track(target_init_pos[0], target_init_pos[1], 1005, ((-10, 10), (-10, 10)), poly_list)
traj = tg.gen_random_rrt_track(0, 0, 1050, ((-10, 10), (-10, 10)), poly_list)
traj = np.array(traj)

# predictor initialization
tau = 30
pred_len = 150
kf_predictor = KFPredictor(tau, pred_len)

images = []
ita = 0
MSELoss_list = []
while ita < traj.__len__() - pred_len:
    # predict trajectory
    observe_start = max(0, ita - tau + 1)
    traj_observed = traj[observe_start: ita + 1]
    pred_traj = kf_predictor.predict(traj_observed, ita)
    pred_start = ita + 1
    # compute MSE Loss
    if ita <= len(traj) - pred_len - tau:
        MSELoss = np.square(np.subtract(traj[pred_start: pred_start + pred_len], pred_traj)).mean()
        MSELoss_list.append(MSELoss)
        print('MSE Loss: ', MSELoss)
    # plot
    # image1, = ax.plot(traj[: pred_start, 0], traj[: pred_start, 1], color='skyblue', linewidth=1, marker='o',
    #                   markersize=1.6)
    image2, = ax.plot(traj_observed[:, 0], traj_observed[:, 1], color='skyblue',
                      linewidth=1, marker='o', markersize=1.6)
    image3, = ax.plot(traj[pred_start: pred_start + pred_len, 0], traj[pred_start: pred_start + pred_len, 1],
                      color='lightgreen', linewidth=1, marker='o', markersize=1.6)
    image4, = ax.plot(pred_traj[:, 0], pred_traj[:, 1], color='red', linewidth=0.5, marker='o', markersize=0.8)
    # plot target
    target = patches.RegularPolygon((traj[pred_start, 0], traj[pred_start, 1]), 4, 0.2 * np.sqrt(2),
                                    color='tomato')
    target_img = ax.add_patch(target)
    # plot selected goal
    images.append([image2, image3, image4, target_img])

    ita += 1

MSE_total = np.array(MSELoss_list)
print('ADE: ', MSE_total.mean())
print('Max-MSE: ', MSE_total.max())
print('Min-MSE: ', MSE_total.min())

ani = animation.ArtistAnimation(fig, images, interval=50, blit=False)
plt.show()