import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import geopandas as gpd
import sys
sys.path.append('../../')
from evaluation.eval_simulation.utils import gen_test_env_poly_list_env


def analyze_run(data):
    """
    Analyze success rate, path distance, path time, and path avg spd
    :param data: run_data
    :return: state_list, path_dis, path_time, path_spd
    """
    run_num = len(data["final_state"])
    state_list = [0, 0, 0]
    path_dis = np.zeros(run_num)
    path_time = np.zeros(run_num)
    # avg_time = 0
    path_spd = np.zeros(run_num)
    for r in range(run_num):
        if data["final_state"][r] == 1 or data["final_state"][r] == 3:
            tmp_overll_path_dis = 0
            for d in range(len(data["robot_path"][r]) - 1):
                rob_pos = data["robot_path"][r][d]
                next_rob_pos = data["robot_path"][r][d + 1]
                tmp_dis = np.sqrt((next_rob_pos[0] - rob_pos[0]) ** 2 + (next_rob_pos[1] - rob_pos[1]) ** 2)
                tmp_overll_path_dis += tmp_dis
            path_dis[r] = tmp_overll_path_dis
            path_time[r] = data["time"][r]
            path_spd[r] = path_dis[r] / path_time[r]
        if data["final_state"][r] == 1:
            state_list[0] += 1
        elif data["final_state"][r] == 2:
            state_list[1] += 1
        elif data["final_state"][r] == 3:
            # print(r)
            state_list[2] += 1
            # avg_time = ((state_list[2]-1) / state_list[2]) * avg_time + path_time[r]/state_list[2]
            # print("average time of an episode: ", avg_time)
        else:
            print("FINAL STATE TYPE ERROR ...")
    return run_num, state_list, path_dis, path_time, path_spd


def plot_robot_goal_paths(data1, data2, poly_list, env_size=((-10, 10), (-10, 10))):
    """
    Plot Robot Path and Goal Path from experiment
    """
    robot_path1 = data1["robot_path"]
    target_path1 = data1["target_path"]
    final_state1 = data1["final_state"]
    robot_path2 = data2["robot_path"]
    target_path2 = data2["target_path"]
    final_state2 = data2["final_state"]

    # good comparison(DDPG SP-DDPG DDPG-GM pH-DDPG-3):
    # 43(bad-good-bad-good)
    # 39(bad-bad-good-good)
    # 33(bad-bad-bad-good)
    k = 33
    robot_p1 = robot_path1[k]
    robot_p1_x = [robot_p1[num][0] for num in range(len(robot_p1))]
    robot_p1_y = [robot_p1[num][1] for num in range(len(robot_p1))]
    target_p1 = target_path1[k]
    del(target_p1[0])
    target_p1_x = [target_p1[num][0] for num in range(len(target_p1))]
    target_p1_y = [target_p1[num][1] for num in range(len(target_p1))]

    robot_p2 = robot_path2[k]
    robot_p2_x = [robot_p2[num][0] for num in range(len(robot_p2))]
    robot_p2_y = [robot_p2[num][1] for num in range(len(robot_p2))]
    target_p2 = target_path2[k]
    del(target_p2[0])
    target_p2_x = [target_p2[num][0] for num in range(len(target_p2))]
    target_p2_y = [target_p2[num][1] for num in range(len(target_p2))]

    if final_state2[k] == -1:
        print("Wrong Final State Value ...")
    else:
        fig, ax = plt.subplots(1, 2, figsize=(28, 20))
        # plot obstacles
        for i, obs in enumerate(poly_list):
            if i > 0:
                p = gpd.GeoSeries(obs)
                p.plot(color='k', ax=ax[0])
                p.plot(color='k', ax=ax[1])
            else:
                ax[0].plot(*obs.xy, 'k-')
                ax[1].plot(*obs.xy, 'k-')

        # plot paths
        ax[0].plot(robot_p1_x, robot_p1_y, color='#4169E1', linestyle='-', lw=1)
        ax[0].scatter(robot_p1_x[0], robot_p1_y[0], color='b', s=30)
        ax[0].scatter(robot_p1_x[-1], robot_p1_y[-1], color='b', s=90)
        ax[0].plot(target_p1_x, target_p1_y, color='r', linestyle='-', lw=1)
        ax[0].scatter(target_p1_x[0], target_p1_y[0], color='r', s=30)
        target = patches.RegularPolygon((target_p1_x[-1], target_p1_y[-1]), 4, 0.2 * np.sqrt(2), color='red')
        ax[0].add_patch(target)
        ax[0].set_xlim(env_size[0])
        ax[0].set_ylim(env_size[1])
        ax[0].set_aspect('equal', 'box')
        ax[0].set_title("DDPG-GM", fontsize=50, y=1.03)
        # ax[0].set_ylabel("Case 3", fontsize=50)
        # rotate ylabel
        labels = ax[0].get_yticklabels()
        plt.setp(labels, rotation=90, horizontalalignment='right')

        ax[1].plot(robot_p2_x, robot_p2_y, color='#4169E1', linestyle='-', lw=1)
        ax[1].scatter(robot_p2_x[0], robot_p2_y[0], color='b', s=30)
        ax[1].scatter(robot_p2_x[-1], robot_p2_y[-1], color='b', s=90)
        ax[1].plot(target_p2_x, target_p2_y, color='r', linestyle='-', lw=1)
        ax[1].scatter(target_p2_x[0], target_p2_y[0], color='r', s=30)
        target = patches.RegularPolygon((target_p2_x[-1], target_p2_y[-1]), 4, 0.2 * np.sqrt(2), color='red')
        ax[1].add_patch(target)
        ax[1].set_xlim(env_size[0])
        ax[1].set_ylim(env_size[1])
        ax[1].set_aspect('equal', 'box')
        ax[1].set_title("pH-DDPG", fontsize=50, y=1.03)

        # Remove x, y Ticks
        ax[0].xaxis.set_ticks_position('none')
        ax[0].yaxis.set_ticks_position('none')
        ax[1].xaxis.set_ticks_position('none')
        ax[1].yaxis.set_ticks_position('none')
        ax[0].xaxis.set_ticklabels([])
        ax[0].yaxis.set_ticklabels([])
        ax[1].xaxis.set_ticklabels([])
        ax[1].yaxis.set_ticklabels([])


if __name__ == "__main__":
    # plot navigation routes
    # MODEL_NAME1 = 'DDPG'
    # MODEL_NAME2 = 'SP-DDPG'
    # MODEL_NAME1 = 'DDPG-GM'
    MODEL_NAME1 = 'NaiveDDPG-Sp-2'
    # MODEL_NAME2 = 'xxx_randomtrain_ctrlstep5_distreward'
    # MODEL_NAME1 = 'ddpg_short_pred_N25'
    # MODEL_NAME2 = 'pH-DDPG-0_simple_track'
    # MODEL_NAME2 = 'pH-DDPG-0'
    # MODEL_NAME2 = 'pH-DDPG-1'
    # MODEL_NAME1 = 'pH-DDPG-2'
    # MODEL_NAME2 = 'pH-DDPG-3'
    # MODEL_NAME1 = 'pH-DDPG-4'
    # MODEL_NAME2 = 'pH-DDPG-5'
    # MODEL_NAME2 = 'xxx_hyperparam2_eval'
    # MODEL_NAME2 = 'xxx_hyperparam2_plus'
    FILE_NAME1 = MODEL_NAME1 + '_0_199_simple.p'
    # FILE_NAME1 = MODEL_NAME1 + '_0_199.p'
    # FILE_NAME2 = MODEL_NAME1 + '_0_199_simple.p'
    FILE_NAME2 = MODEL_NAME1 + '_0_199.p'
    # FILE_NAME1 = 'pH-DDPG-1-5_0_999_simple.p'
    # FILE_NAME2 = 'pH-DDPG-1-5_0_999.p'

    run_data1 = pickle.load(open('../record_data/' + FILE_NAME1, 'rb'))
    run_num, termination, path_dist, path_time, path_spd = analyze_run(run_data1)
    path_dist_avg = np.mean(path_dist[path_dist > 0])
    path_dist_std = np.std(path_dist[path_dist > 0])
    dist_CI_width = 1.95996 * path_dist_std / np.sqrt(run_num)
    path_time_avg = np.mean(path_time[path_dist > 0])
    path_time_std = np.std(path_time[path_dist > 0])
    time_CI_width = 1.95996 * path_time_std / np.sqrt(run_num)
    print(FILE_NAME1 + " random simulation results:")
    print("Success: ", termination[0], " Collision: ", termination[1], " Overtime: ", termination[2])
    print("95% CI of Path Distance of Success and Overtime Routes: ", path_dist_avg, '±', dist_CI_width, ' m')
    print("95% CI of Path Time of Success and Overtime Routes: ", path_time_avg, '±', time_CI_width, ' s')
    print("Average Path Speed of Success and Overtime Routes: ", np.mean(path_spd[path_dist > 0]), ' m/s')
    print('\n')

    run_data2 = pickle.load(open('../record_data/' + FILE_NAME2, 'rb'))
    run_num, termination, path_dist, path_time, path_spd = analyze_run(run_data2)
    path_dist_avg = np.mean(path_dist[path_dist > 0])
    path_dist_std = np.std(path_dist[path_dist > 0])
    dist_CI_width = 1.95996 * path_dist_std / np.sqrt(run_num)
    path_time_avg = np.mean(path_time[path_dist > 0])
    path_time_std = np.std(path_time[path_dist > 0])
    time_CI_width = 1.95996 * path_time_std / np.sqrt(run_num)
    print(FILE_NAME2 + " random simulation results:")
    print("Success: ", termination[0], " Collision: ", termination[1], " Overtime: ", termination[2])
    print("95% CI of Path Distance of Success and Overtime Routes: ", path_dist_avg, '±', dist_CI_width, ' m')
    print("95% CI of Path Time of Success and Overtime Routes: ", path_time_avg, '±', time_CI_width, ' s')
    print("Average Path Speed of Success and Overtime Routes: ", np.mean(path_spd[path_dist > 0]), ' m/s')
    # poly_list, _ = gen_test_env_poly_list_env()
    # plt.rc('font', family='Times New Roman')
    # plot_robot_goal_paths(run_data1, run_data2, poly_list)
    # plt.show()
