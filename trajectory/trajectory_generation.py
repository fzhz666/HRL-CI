import numpy as np
import math
import random
from collections import deque
from shapely.geometry import Point, LineString
from trajectory.rrt import RRT


def is_saw(x, y, n):
    '''
    Checks if walk of length n is self-avoiding
    :param (x,y)(list,list): walk of length n
    :param n(int): length of the walk
    :return: True if the walk is self-avoiding
    '''
    # creating a set removes duplicates, so it suffices to check the size of the set
    return n+1 == len(set(zip(x, y)))


def in_map(x, y):
    '''
    Checks if walk is in map
    :param (x,y)(list,list): the walk
    :return judge: True if the walk is in map
    '''
    judge = False
    x, y = np.array(x), np.array(y)
    if (abs(x) < 9.5).all() and (abs(y) < 9.5).all():
        judge = True
    return judge


def myopic_saw(n, x_init, y_init):
    '''
    Tries to generate a SAW of length n using myopic algorithm
    :param n(int): length of walk
    :param (x_init, y_init)(float, float): initial coordinate of saw
    :return (x,y)(list,list): the walk (length <= n)
    '''
    theta = [i * math.pi / 2 for i in range(0, 4)]
    r = 5    # original one: 0.75
    x, y = [x_init], [y_init]
    positions = set([(x_init,y_init)])    # positions is a set(no same element) that stores all sites visited
    for i in range(n):
        t = random.choice(theta)
        x_new = x[-1] + r * round(math.cos(t), 2)
        y_new = y[-1] + r * round(math.sin(t), 2)
        if (x_new, y_new) not in positions:
            x.append(x_new)
            y.append(y_new)
            positions.add((x_new, y_new))
        else:
            continue
    return x, y


def dimer(n, x_init, y_init):
    '''
    Generates a SAW by dimerization
    :param n(int): length of walk
    :param (x_init, y_init)(float, float): initial coordinate of saw
    :return (x_concat,y_concat)(list,list): walk of length n
    '''
    if n <= 3:
        x, y = myopic_saw(n, x_init, y_init)
        return x, y
    else:
        not_saw = True
        while not_saw:
            x_1, y_1 = dimer(n//2, x_init, y_init)
            x_2, y_2 = dimer(n - n//2, x_init, y_init)
            x_2 = [(x_1[-1] - x_1[0] + x) for x in x_2]
            y_2 = [(y_1[-1] - y_1[0] + y) for y in y_2]
            x_concat, y_concat = x_1 + x_2[1:], y_1 + y_2[1:]
            if is_saw(x_concat, y_concat, n) and in_map(x_concat, y_concat):
                not_saw = False
        return x_concat, y_concat


def interpolation(x, y, n_trajectory):
    '''
    Generates continous trajectory by whittaker-shannon interpolation formula
    :param (x,y)(list,list): discrete self-avoiding walk
    :param n_trajectory(int): length of continous trajectory
    :return (x_contin,y_contin)(list,list): continous trajectory
    '''
    interval = 1
    N = len(x)
    t = np.linspace(0, (N - 1) * interval, n_trajectory)
    x_contin, y_contin = 0, 0
    for n in range(N):
        x_contin += x[n] * np.sinc(t / interval - n)
        y_contin += y[n] * np.sinc(t / interval - n)
    return x_contin, y_contin


def bezier_smooth(traj_points, n_trajectory):
    raw_trajectory = np.array(traj_points)
    raw_traj_x = raw_trajectory[:, 0]
    raw_traj_y = raw_trajectory[:, 1]

    N = raw_trajectory.shape[0] - 1
    t = np.linspace(0, 1, n_trajectory)
    polynomial = []
    for i in range(N+1):
        polynomial.append(math.comb(N, i) * (t ** (N - i)) * (1 - t) ** i)

    polynomial = np.array(polynomial)
    traj_x = np.dot(raw_traj_x, polynomial)
    traj_y = np.dot(raw_traj_y, polynomial)
    return list(zip(traj_x, traj_y))


def gen_saw_track(x_init, y_init, n_trajectory=1100, n_discre=69):
    '''
    Generates continous self-avoiding trajectory
    :param (x_init, y_init)(float, float): initial coordinate of saw
    :param n_discre(int): length of discrete self-avoiding walk
    :param n_trajectory(int): length of continous trajectory
    :return trajectory(list): continous trajectory
    '''
    x, y = dimer(n_discre, x_init, y_init)
    trajectory = bezier_smooth(list(zip(x, y)), n_trajectory)
    return trajectory


def gen_rose_track(x_init, y_init, n_trajectory=1200):
    x, y = [], []
    for i in range(n_trajectory):
        k = i * np.pi / 400
        r = 5 * np.sin(4 * k)
        x.append(0.5 * x_init + r * np.cos(k))
        y.append(0.5 * y_init + r * np.sin(k))
    trajectory = list(zip(x, y))
    return trajectory


def gen_spiral_track(x_init, y_init, n_trajectory=1200):
    x, y = [], []
    for i in range(n_trajectory):
        k = i * math.pi / 90
        r = 0.12 * k
        x.append(0.5 * x_init + r * math.cos(k))
        y.append(0.5 * y_init + r * math.sin(k))
    trajectory = list(zip(x, y))
    return trajectory


def gen_circle_track(x_init, y_init, r=10, n_trajectory=1010):
    x, y = [], []
    for i in range(n_trajectory):
        k = i * math.pi / 90
        x.append(x_init + r * math.cos(k))
        y.append(y_init + r * math.sin(k))
    trajectory = list(zip(x, y))
    return trajectory


def gen_square_track(x_init, y_init, a=19, r=6):
    '''
    number of trajectory points is 1012
    '''
    x, y = [], []
    i = 0
    center = [x_init+(a/2-r), y_init+(a/2-r)]
    while i < 150:
        k = i * math.pi / 300
        x.append(center[0] + r * math.cos(k))
        y.append(center[1] + r * math.sin(k))
        i += 1
    for j in range(103):
        x.append(x[-1] - (a-2*r)/100)
        y.append(y[-1])

    center = [x_init-(a/2-r), y_init+(a/2-r)]
    while i < 300:
        k = i * math.pi / 300
        x.append(center[0] + r * math.cos(k))
        y.append(center[1] + r * math.sin(k))
        i += 1
    for j in range(103):
        x.append(x[-1])
        y.append(y[-1] - (a-2*r)/100)

    center = [x_init-(a/2-r), y_init-(a/2-r)]
    while i < 450:
        k = i * math.pi / 300
        x.append(center[0] + r * math.cos(k))
        y.append(center[1] + r * math.sin(k))
        i += 1
    for j in range(103):
        x.append(x[-1] + (a-2*r)/100)
        y.append(y[-1])

    center = [x_init+(a/2-r), y_init-(a/2-r)]
    while i < 600:
        k = i * math.pi / 300
        x.append(center[0] + r * math.cos(k))
        y.append(center[1] + r * math.sin(k))
        i += 1
    for j in range(103):
        x.append(x[-1])
        y.append(y[-1] + (a-2*r)/100)

    trajectory = list(zip(x, y))
    return trajectory


def gen_random_rrt_track(x_init, y_init, n_trajectory, env_size, poly_list):
    p_start = [x_init, y_init]
    step_len = 0.5
    random_path_len = 11   # eval-random: 11-100-400    eval-simple: 18-200-400
    self_avoid_poly = deque([Point(-10, -10), Point(-10, -10), Point(-10, -10)], maxlen=3)
    poly_list.extend(self_avoid_poly)
    overall_path = []
    while len(overall_path) < 400:  # 400
        # 每次新增的路径点数为random_path_len，random增加的少，最终新增路径的长度更长，而步数都差不多，故random的更快？
        rrt = RRT(p_start, (0, 0), step_len, random_path_len=random_path_len)
        random_path = rrt.random_planning(env_size, poly_list)
        if not random_path:
            return None
        new_path = bezier_smooth(random_path, 100)  # 100
        overall_path.extend(new_path)

        self_avoid_poly.append(LineString(new_path[:-10]))
        poly_list[-3:] = self_avoid_poly
        p_start = overall_path[-1]

    trajectory = bezier_smooth(overall_path, n_trajectory)
    trajectory.reverse()
    return trajectory


def gen_simple_rrt_track(multi_goal_list, n_trajectory, env_size, poly_list):
    step_len = 0.5
    rrt = RRT(multi_goal_list[0], multi_goal_list[1], step_len)
    path1 = rrt.path_planning(env_size, poly_list)
    rrt = RRT(multi_goal_list[1], multi_goal_list[2], step_len)
    path2 = rrt.path_planning(env_size, poly_list)
    if not path1 or not path2:
        return None
    overall_path = bezier_smooth(path1, 200)
    overall_path.extend(bezier_smooth(path2, 200))
    trajectory = bezier_smooth(overall_path, n_trajectory)
    trajectory.reverse()
    return trajectory
