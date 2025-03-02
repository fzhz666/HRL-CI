import pickle
import numpy as np
import matplotlib.pyplot as plt
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
    path_spd = np.zeros(run_num)
    for r in range(run_num):
        if data["final_state"][r] == 1:
            state_list[0] += 1
            tmp_overll_path_dis = 0
            for d in range(len(data["path"][r]) - 1):
                rob_pos = data["path"][r][d]
                next_rob_pos = data["path"][r][d + 1]
                tmp_dis = np.sqrt((next_rob_pos[0] - rob_pos[0]) ** 2 + (next_rob_pos[1] - rob_pos[1]) ** 2)
                tmp_overll_path_dis += tmp_dis
            path_dis[r] = tmp_overll_path_dis
            path_time[r] = data["time"][r]
            path_spd[r] = path_dis[r] / path_time[r]
        elif data["final_state"][r] == 2:
            state_list[1] += 1
        elif data["final_state"][r] == 3:
            state_list[2] += 1
        else:
            print("FINAL STATE TYPE ERROR ...")
    return state_list, path_dis, path_time, path_spd


def plot_robot_paths(data, poly_list, goal_list, env_size=((-10, 10), (-10, 10))):
    """
    Plot Robot Path from experiment
    :param path: robot path
    :param final_state: final states
    :param poly_list: obstacle poly list
    """
    path = data["path"]
    final_state = data["final_state"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    for obs in poly_list:
        x = [obs[num][0] for num in range(len(obs))]
        y = [obs[num][1] for num in range(len(obs))]
        x.append(obs[0][0])
        y.append(obs[0][1])
        ax[0].plot(x, y, 'k-')
        ax[1].plot(x, y, 'k-')
    for i, p in enumerate(path, 0):
        p_x = [p[num][0] for num in range(len(p))]
        p_y = [p[num][1] for num in range(len(p))]
        if final_state[i] == 1:
            ax[0].plot(p_x, p_y, color='#4169E1', linestyle='-', lw=0.5)
            ax[0].plot([p_x[0]], [p_y[0]], 'bo')
            ax[0].plot([p_x[-1]], [p_y[-1]], 'ro')
        elif final_state[i] == 2:
            ax[1].plot(p_x, p_y, color='#4169E1', linestyle='-', lw=0.8)
            ax[1].plot([p_x[0]], [p_y[0]], 'bo')
            ax[1].plot([p_x[-1]], [p_y[-1]], 'rx')
            ax[1].plot([goal_list[i][0]], [goal_list[i][1]], 'ro')
            ax[1].plot([p_x[-1], goal_list[i][0]], [p_y[-1], goal_list[i][1]], 'r--', lw=0.8)
        elif final_state[i] == 3:
            ax[1].plot(p_x, p_y, color='#4169E1', linestyle='-', lw=0.8)
            ax[1].plot([p_x[0]], [p_y[0]], 'bo')
            ax[1].plot([p_x[-1]], [p_y[-1]], 'go')
            ax[1].plot([goal_list[i][0]], [goal_list[i][1]], 'ro')
            ax[1].plot([p_x[-1], goal_list[i][0]], [p_y[-1], goal_list[i][1]], 'r--', lw=0.8)
        else:
            print("Wrong Final State Value ...")
    ax[0].set_xlim(env_size[0])
    ax[0].set_ylim(env_size[1])
    ax[0].set_aspect('equal', 'box')
    ax[0].set_title("Success Routes")
    ax[1].set_xlim(env_size[0])
    ax[1].set_ylim(env_size[1])
    ax[1].set_aspect('equal', 'box')
    ax[1].set_title("Failure Routes (Collision + Overtime)")


MODEL_NAME = 'sddpg_bw_5'
# MODEL_NAME = 'pH-DDPG-5'

# MODEL_NAME = 'ddpg'
# MODEL_NAME = 'ddpg_poisson'
FILE_NAME = MODEL_NAME + '_0_199.p'
# FILE_NAME = MODEL_NAME + '_0_199_simple.p'

run_data = pickle.load(open('../record_data/' + FILE_NAME, 'rb'))
s_list, p_dis, p_time, p_spd = analyze_run(run_data)
print(MODEL_NAME + " random simulation results:")
print("Success: ", s_list[0], " Collision: ", s_list[1], " Overtime: ", s_list[2])
print("Average Path Distance of Success Routes: ", np.mean(p_dis[p_dis > 0]), ' m')
print("Average Path Time of Success Routes: ", np.mean(p_time[p_dis > 0]), ' s')
print("Average Path Speed of Success Routes: ", np.mean(p_spd[p_dis > 0]), ' m/s')


start_goal_pos = pickle.load(open("../eval_simulation/eval_positions.p", "rb"))
goal_list = start_goal_pos[1]
poly_list, raw_poly_list = gen_test_env_poly_list_env()
plot_robot_paths(run_data, raw_poly_list, goal_list)
plt.show()
