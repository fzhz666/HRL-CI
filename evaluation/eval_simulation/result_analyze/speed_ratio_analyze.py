# Compute the average & maximum speed of target with motion policy 1 & 2
import numpy as np
import pickle


def analyze_spd(paths_list):
    """
    Compute path avg spd & max spd of target
    :param paths_list: run_data
    :return: state_list, path_dis, path_time, path_spd
    """
    run_num = len(paths_list)
    path_dis = np.zeros(run_num)
    time_episode = 99.9147742064794
    path_avg_spd = np.zeros(run_num)
    path_max_spd = np.zeros(run_num)
    path_min_spd = np.zeros(run_num)
    for r in range(run_num):
        tmp_overll_path_dis = 0
        tmp_dis_array = np.zeros(999)
        for d in range(999):
            pos = paths_list[r][d]
            next_pos = paths_list[r][d + 1]
            tmp_dis = np.sqrt((next_pos[0] - pos[0]) ** 2 + (next_pos[1] - pos[1]) ** 2)
            tmp_dis_array[d] = tmp_dis
            tmp_overll_path_dis += tmp_dis
        path_dis[r] = tmp_overll_path_dis
        path_avg_spd[r] = path_dis[r] / time_episode
        path_max_spd[r] = np.max(tmp_dis_array)/time_episode * 1000
        path_min_spd[r] = np.min(tmp_dis_array)/time_episode * 1000
    avg_spd = np.mean(path_avg_spd)
    max_spd = np.mean(path_max_spd)
    min_spd = np.mean(path_min_spd)
    return avg_spd, max_spd, min_spd


if __name__ == "__main__":
    simple_paths_robot_list = pickle.load(open('../eval_simulation/eval_simple_paths_robot_pose.p', 'rb'))
    rand_paths_robot_list = pickle.load(open('../eval_simulation/eval_rand_paths_robot_pose.p', 'rb'))
    simple_target_paths_list = simple_paths_robot_list[1][:]
    rand_target_paths_list = rand_paths_robot_list[1][:]
    simple_avg_spd, simple_max_spd, simple_min_spd = analyze_spd(simple_target_paths_list)
    rand_avg_spd, rand_max_spd, rand_min_spd = analyze_spd(rand_target_paths_list)
    robot_avg_spd_all = np.array([0.43881771129936054, 0.43881771129936054, 0.40736484096490094, 0.4253098432347462,
                                  0.434450803294224, 0.4262795863852764, 0.43284849128621816, 0.4250617281264676,
                                  0.43924654013162867, 0.4328438735721875, 0.405880745449911, 0.4065695220692425,
                                  0.4107411041551204, 0.40826960229507897, 0.4217317771925163, 0.4048474236935918])
    robot_avg_spd_simple = np.mean(robot_avg_spd_all[:8])
    robot_avg_spd_rand = np.mean(robot_avg_spd_all[8:])
    robot_max_spd = 0.5
    robot_min_spd = 0.05
    print("Average Path Speed of with Motion Policy 1\nTarget: ", simple_avg_spd, ' m/s, '
          'Robot: ', robot_avg_spd_simple, 'm/s')
    print("Maximum Path Speed of with Motion Policy 1\nTarget: ", simple_max_spd, ' m/s, '
          'Robot: ', robot_max_spd, 'm/s')
    print("Minimum Path Speed of with Motion Policy 1\nTarget: ", simple_min_spd, ' m/s, '
          'Robot: ', robot_min_spd, 'm/s')
    print('\n')
    print("Average Path Speed of with Motion Policy 2\nTarget: ", rand_avg_spd, ' m/s, '
          'Robot: ', robot_avg_spd_rand, 'm/s')
    print("Maximum Path Speed of with Motion Policy 2\nTarget: ", rand_max_spd, ' m/s, '
          'Robot: ', robot_max_spd, 'm/s')
    print("Minimum Path Speed of with Motion Policy 2\nTarget: ", rand_min_spd, ' m/s, '
          'Robot: ', robot_min_spd, 'm/s')
