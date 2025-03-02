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


def gen_sample_map_poly_list():
    """
    Generate Poly list of test environment
    :return: poly_list
    """
    env = [(-10, 10), (10, 10), (10, -10), (-10, -10)]
    obs2 = [(-5.5, -6), (-5, -6), (-5, -10), (-5.5, -10)]
    obs3 = [(-3, -5), (4, -5), (4, -6), (-3, -6)]
    obs5 = [(-6, -2), (-5, -2), (-5, -3), (-6, -3)]
    obs6 = [(-9.25, 1), (-4, 1), (-4, 0), (-9.25, 0)]
    obs11 = [(4.5, 7.5), (5.5, 8.5), (6.5, 7.5), (5.5, 6.5)]
    obs13 = [(-6, 6), (-7, 6), (-7, 5), (-6, 5)]
    obs14 = [(5, 4), (5.5, 4), (5.5, 0), (5, 0)]
    obs15 = [(5.5, 4), (9, 4), (9, 3.5), (5.5, 3.5)]
    obs20 = [(-3, 6), (1.5, 6), (1.5, 2.5), (-3, 2.5)]
    obs22 = [(0, -3), (6, -3), (6, -1), (0, -1)]
    poly_raw_list = [env, obs2, obs3, obs6, obs5,
                     obs11, obs13, obs14, obs15,
                     obs20, obs22]
    poly_list = utils.gen_polygon_exterior_list(poly_raw_list)
    return poly_list, poly_raw_list


# plot obstacles
import geopandas as gpd
fig, ax = plt.subplots(figsize=(10, 10))

poly_list, _ = gen_sample_map_poly_list()
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
while ita < traj.__len__() - pred_len:
    # predict trajectory
    observe_start = max(0, ita - tau + 1)
    traj_observed = traj[observe_start: ita + 1]
    pred_traj = kf_predictor.predict(traj_observed, ita)
    pred_start = ita + 1
    # plot
    # image1, = ax.plot(traj[: pred_start, 0], traj[: pred_start, 1], color='skyblue', linewidth=1, marker='o',
    #                   markersize=1.6)
    image2, = ax.plot(traj_observed[:, 0], traj_observed[:, 1], color='skyblue',
                      linewidth=1, marker='o', markersize=1.6)
    # image3, = ax.plot(traj[pred_start: pred_start + pred_len, 0], traj[pred_start: pred_start + pred_len, 1],
    #                   color='lightgreen', linewidth=1, marker='o', markersize=1.6)
    # image4, = ax.plot(pred_traj[:, 0], pred_traj[:, 1], color='red', linewidth=0.5, marker='o', markersize=0.8)
    # plot target
    target = patches.RegularPolygon((traj[pred_start, 0], traj[pred_start, 1]), 4, 0.2 * np.sqrt(2),
                                    color='tomato')
    target_img = ax.add_patch(target)
    # plot selected goal
    images.append([image2, target_img])

    ita += 1

ani = animation.ArtistAnimation(fig, images, interval=50, blit=False)
plt.show()