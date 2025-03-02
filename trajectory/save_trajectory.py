import numpy as np
import random
import pickle
from copy import deepcopy
from shapely.geometry import Point
import trajectory_generation as tg
from evaluation.eval_simulation.utils import *
from training.utils import *


def gen_init_position_list(poly_list, env_size=((-7, 7), (-7, 7)), obs_near_th=0.5, sample_step=0.1):
    """
    Generate list of goal positions
    :param poly_list: list of obstacle polygon
    :param env_size: size of the environment
    :param obs_near_th: Threshold for near an obstacle
    :param sample_step: sample step for goal generation
    :return: goal position list
    """
    goal_pos_list = []
    x_pos, y_pos = np.mgrid[env_size[0][0]:env_size[0][1]:sample_step, env_size[1][0]:env_size[1][1]:sample_step]
    for x in range(x_pos.shape[0]):
        for y in range(x_pos.shape[1]):
            tmp_pos = [x_pos[x, y], y_pos[x, y]]
            tmp_point = Point(tmp_pos[0], tmp_pos[1])
            near_obstacle = False
            for poly in poly_list:
                tmp_dis = tmp_point.distance(poly)
                if tmp_dis < obs_near_th:
                    near_obstacle = True
            if near_obstacle is False:
                goal_pos_list.append(tmp_pos)
    return goal_pos_list


def gen_robot_pose_n_target(all_positions, robot_goal_diff=9):
    """
    Randomly generate robot pose and goal
    """
    target = random.choice(all_positions)
    pose = random.choice(all_positions)
    dist = np.hypot(target[0] - pose[0], target[1] - pose[1])
    while dist < robot_goal_diff:
        pose = random.choice(all_positions)
        dist = np.hypot(target[0] - pose[0], target[1] - pose[1])
    pose.append(random.random() * 2 * np.pi)
    return pose, target


def save_list(list, file_path):
    f = open(file_path, 'wb')
    pickle.dump(list, f)
    f.close()


def load_list(file_path):
    f = open(file_path, 'rb')
    list = pickle.load(f)
    f.close()
    return list


def trajectory_plot(poly_list, positions_all):
    import matplotlib.pyplot as plt
    import geopandas as gpd

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, obs in enumerate(poly_list):
        if i > 0:
            p = gpd.GeoSeries(obs)
            p.plot(color='k', ax=ax)
        else:
            ax.plot(*obs.xy, 'k-')
    ax.set_xticks(np.linspace(-13, 13, 27))
    ax.set_yticks(np.linspace(17, 43, 27))

    # plot random RRT path
    # robo_pose, target = gen_robot_pose_n_target(positions_all, robot_goal_diff=9)
    # traj = tg.gen_random_rrt_track(target[0], target[1], 1005,
                                   # env_size=((-11, 11), (22, 41)),
                                   # env_size=((-10, 10), (-10, 10)),
                                   # poly_list=deepcopy(poly_list))

    # plot simple RRT path
    multi_goal_list = random.sample([[-11, 41], [11, 41], [11, 22], [-11, 22]], 3)
    robo_pose = random.choice(positions_all)
    dist = np.hypot(multi_goal_list[0][0] - robo_pose[0], multi_goal_list[0][1] - robo_pose[1])
    while dist < 16:
        robo_pose = random.choice(positions_all)
        dist = np.hypot(multi_goal_list[0][0] - robo_pose[0], multi_goal_list[0][1] - robo_pose[1])
    robo_pose.append(random.random() * 2 * np.pi)
    traj = tg.gen_simple_rrt_track(multi_goal_list, 1020, env_size=((-11, 11), (22, 41)),
                                   poly_list=deepcopy(train_poly_list))
    ax.scatter(multi_goal_list[0][0], multi_goal_list[0][1], s=40, c='r')
    ax.scatter(multi_goal_list[1][0], multi_goal_list[1][1], s=40, c='r')
    ax.scatter(multi_goal_list[2][0], multi_goal_list[2][1], s=40, c='r')
    ax.text(multi_goal_list[0][0], multi_goal_list[0][1], '1', fontsize=15)
    ax.text(multi_goal_list[1][0], multi_goal_list[1][1], '2', fontsize=15)
    ax.text(multi_goal_list[2][0], multi_goal_list[2][1], '3', fontsize=15)

    ax.text(traj[0][0], traj[0][1], 'path start', fontsize=10)
    # ax.scatter(target[0], target[1], s=40, c='r')
    ax.scatter(robo_pose[0], robo_pose[1], s=40, c='b')
    ax.plot([p[0] for p in traj], [p[1] for p in traj], color='red', linewidth=0.5, marker='o',
            markersize=0.8)
    plt.show()


def gen_training_paths(poly_list, positions_all, pathtype='simple'):
    save_flag = False
    robot_pose_list = []
    target_paths_list = []
    target_path = None
    robot_pose = None
    if pathtype == 'simple':
        robot_goal_diff = 16
        for i in range(500):
            j = 0
            while not target_path:
                multi_goal_list = random.sample([[-11, 41], [11, 41], [11, 22], [-11, 22]], 3)
                print('generating initial target position...')
                robot_pose = random.choice(positions_all)
                dist = np.hypot(multi_goal_list[0][0] - robot_pose[0], multi_goal_list[0][1] - robot_pose[1])
                while dist < robot_goal_diff:
                    robot_pose = random.choice(positions_all)
                    dist = np.hypot(multi_goal_list[0][0] - robot_pose[0], multi_goal_list[0][1] - robot_pose[1])
                robot_pose.append(random.random() * 2 * np.pi)
                print('path generating attemption ', j + 1)
                target_path = tg.gen_simple_rrt_track(multi_goal_list, 1010, env_size=((-11, 11), (22, 41)),
                                                      poly_list=deepcopy(train_poly_list))
                j += 1
            robot_pose_list.append(robot_pose)
            target_paths_list.append(target_path)
            print('    ' + str(i + 1) + ' path generated')
            target_path = None
    elif pathtype == 'random':
        for i in range(500):
            j = 0
            while not target_path:
                print('generating initial target position...')
                robot_pose, init_target = gen_robot_pose_n_target(positions_all, robot_goal_diff=10)
                print('path generating attemption ', j + 1)
                target_path = tg.gen_random_rrt_track(init_target[0], init_target[1], 1005,
                                                      env_size=((-11, 11), (22, 41)), poly_list=deepcopy(poly_list))
                j += 1
            robot_pose_list.append(robot_pose)
            target_paths_list.append(target_path)
            print('    ' + str(i + 1) + ' path generated')
            target_path = None
    else:
        print("Wrong path type. Path type should only be 'simple' or 'random'.")
        return save_flag

    rand_path_robot_list = [robot_pose_list, target_paths_list]
    # save_list(rand_path_robot_list, file_path='train_paths_robot_pose.p')
    # load_path_list = load_list(file_path='train_paths_robot_pose.p')

    save_list(rand_path_robot_list, file_path='train_paths_robot_pose_new.p')
    load_path_list = load_list(file_path='train_paths_robot_pose_new.p')
    if load_path_list == rand_path_robot_list:
        save_flag = True
    return save_flag


def gen_test_paths(poly_list, positions_all, pathtype='random'):
    '''
    generate target trajectories for testing
    '''
    save_flag = False
    robot_pose_list = []
    target_paths_list = []
    target_path = None
    robot_pose = None
    if pathtype == 'simple':
        robot_goal_diff = 9
        for i in range(200):
            j = 0
            while not target_path:
                init_target = random.choice([[-9, 9], [9, 9], [9, -9], [-9, -9]])
                print('generating initial robot position...')
                robot_pose = random.choice(positions_all)
                dist = np.hypot(init_target[0] - robot_pose[0], init_target[1] - robot_pose[1])
                while dist < robot_goal_diff:  # 测试一下看是否需要加 > 约束
                    robot_pose = random.choice(positions_all)
                    dist = np.hypot(init_target[0] - robot_pose[0], init_target[1] - robot_pose[1])
                robot_pose.append(random.random() * 2 * np.pi)
                print('path generating attemption ', j + 1)
                target_path = tg.gen_random_rrt_track(init_target[0], init_target[1], 1050,  # 1050
                                                      env_size=((-10, 10), (-10, 10)), poly_list=deepcopy(poly_list))
                j += 1
            robot_pose_list.append(robot_pose)
            target_paths_list.append(target_path)
            print('    ' + str(i + 1) + ' path generated')
            target_path = None
    elif pathtype == 'random':
        for i in range(200):
            j = 0
            while not target_path:
                print('generating initial target position...')
                robot_pose, init_target = gen_robot_pose_n_target(positions_all)
                print('path generating attemption ', j + 1)
                target_path = tg.gen_random_rrt_track(init_target[0], init_target[1], 1080,
                                                      env_size=((-10, 10), (-10, 10)), poly_list=deepcopy(poly_list))
                j += 1
            robot_pose_list.append(robot_pose)
            target_paths_list.append(target_path)
            print('    ' + str(i + 1) + ' path generated')
            target_path = None
    else:
        print("Wrong path type. Path type should only be 'single' or 'random'.")
        return save_flag

    rand_path_robot_list = [robot_pose_list, target_paths_list]
    save_list(rand_path_robot_list, file_path='eval_paths_robot_pose.p')
    load_path_list = load_list(file_path='eval_paths_robot_pose.p')
    if load_path_list == rand_path_robot_list:
        save_flag = True
    return save_flag


if __name__ == '__main__':
    # _, train_poly_list, _ = gen_poly_list_env5()
    _, train_poly_list, _ = gen_poly_list_env5_new()
    all_training_positions = gen_init_position_list(train_poly_list, env_size=((-11, 11), (20, 41)))
    test_poly_list, _ = gen_test_env_poly_list_env()
    all_test_positions = gen_goal_position_list(test_poly_list, env_size=((-7, 7), (-7, 7)))

    # check randomly generated trajectory (for tuning hyper-parameters)
    # trajectory_plot(train_poly_list, all_training_positions)
    # trajectory_plot(test_poly_list, all_test_positions)

    # save_result = gen_training_paths(train_poly_list, all_training_positions, pathtype='simple')
    save_result = gen_training_paths(train_poly_list, all_training_positions, pathtype='random')
    # save_result = gen_test_paths(test_poly_list, all_test_positions, pathtype='simple')
    # save_result = gen_test_paths(test_poly_list, all_test_positions, pathtype='random')
    print('-------------------------')
    if save_result:
        print('NNNNNNNNNNNNNNNNNNNice~~')

