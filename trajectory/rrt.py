import numpy as np
import math
from shapely.geometry import Point, LineString, Polygon
from collections import deque


class Node:
    def __init__(self, coordinate):
        self.x = coordinate[0]
        self.y = coordinate[1]
        self.parent = None


class RRT:
    def __init__(self, p_start, p_goal, step_len, random_path_len=10, sample_rate=0.05, max_iter=10000):
        self.step_len = step_len
        self.random_path_len = random_path_len
        self.sample_rate = sample_rate
        self.max_iter = max_iter

        self.p_start = Node(p_start)
        self.p_goal = Node(p_goal)
        self.vertex = [self.p_start]

    def path_planning(self, env_size, poly_list):
        for i in range(self.max_iter):
            node_rand = self.random_state(env_size, self.sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if not self.near_obstacle(node_near, node_new, poly_list):
                self.vertex.append(node_new)
                dis, _ = self.get_distance_angle(node_new, self.p_goal)
                if dis <= self.step_len and not self.near_obstacle(node_new, self.p_goal, poly_list):
                    self.p_goal = self.new_state(node_new, self.p_goal)
                    self.vertex.append(self.p_goal)
                    return self.extract_path(self.p_goal)
        return None

    def random_planning(self, env_size, poly_list):
        for i in range(self.max_iter):
            node_rand = self.random_state(env_size, goal_sample_rate=-1)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if not self.near_obstacle(node_near, node_new, poly_list):
                self.vertex.append(node_new)
                dis, _ = self.get_distance_angle(self.p_start, node_new)
                if dis > self.random_path_len:
                    return self.extract_path(node_new)
        return None

    def random_state(self, env_size, goal_sample_rate, delta=0.2):
        # if np.random.random() > goal_sample_rate:
        #     return Node((np.random.uniform(env_size[0][0] + delta, env_size[0][1] - delta),
        #                  np.random.uniform(env_size[1][0] + delta, env_size[1][1] - delta)))
        if np.random.random() > goal_sample_rate:
            env_range_x = env_size[0][1] - env_size[0][0]
            env_range_y = env_size[1][1] - env_size[1][0]
            return Node((np.random.uniform(env_size[0][0] - env_range_x, env_size[0][1] + env_range_x),
                         np.random.uniform(env_size[1][0] - env_range_y, env_size[1][1] + env_range_y)))
        return self.p_goal

    def new_state(self, node_start, node_end):
        dis, angle = self.get_distance_angle(node_start, node_end)
        dis = min(dis, self.step_len)
        node_new = Node((node_start.x + dis * math.cos(angle),
                         node_start.y + dis * math.sin(angle)))
        node_new.parent = node_start
        return node_new

    @staticmethod
    def nearest_neighbor(node_list, node_rand):
        return node_list[np.argmin([math.hypot(n.x - node_rand.x, n.y - node_rand.y) for n in node_list])]

    @staticmethod
    def extract_path(node_end):
        path = []
        node_now = node_end

        while node_now is not None:
            path.append((node_now.x, node_now.y))
            node_now = node_now.parent

        return path

    @staticmethod
    def get_distance_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    @staticmethod
    def near_obstacle(node_start, node_end, poly_list, obs_near_th=0.4):
        """

        :param poly_list: list of obstacle polygon
        :param obs_near_th: Threshold for near an obstacle
        :return:
        """
        point = Point(node_end.x, node_end.y)
        line = LineString([(node_start.x, node_start.y), (node_end.x, node_end.y)])
        is_near_obstacle = False
        for poly in poly_list:
            dis1 = point.distance(poly)
            dis2 = line.distance(poly)
            if dis1 < obs_near_th or dis2 < obs_near_th:
                is_near_obstacle = True
        return is_near_obstacle


def bezier_smooth(traj_points, n_traj):
    raw_trajectory = np.array(traj_points)
    raw_traj_x = raw_trajectory[:, 0]
    raw_traj_y = raw_trajectory[:, 1]

    N = raw_trajectory.shape[0] - 1
    t = np.linspace(0, 1, n_traj)
    polynomial = []
    for i in range(N+1):
        polynomial.append(math.comb(N, i) * (t ** (N - i)) * (1 - t) ** i)

    polynomial = np.array(polynomial)
    traj_x = np.dot(raw_traj_x, polynomial)
    traj_y = np.dot(raw_traj_y, polynomial)
    return list(zip(traj_x, traj_y))


if __name__ == '__main__':
    from evaluation.eval_simulation import utils
    import matplotlib.pyplot as plt


    fig, ax = plt.subplots(figsize=(10, 10))

    env_size = [(-10, 10), (-10, 10)]
    '''
    Single Path
    '''
    poly_list, poly_raw_list = utils.gen_test_env_poly_list_env()
    for obs in poly_raw_list:
        x = [obs[num][0] for num in range(len(obs))]
        y = [obs[num][1] for num in range(len(obs))]
        x.append(obs[0][0])
        y.append(obs[0][1])
        ax.plot(x, y, 'k-')
    ax.set_xticks(np.linspace(-10, 10, 21))
    ax.set_yticks(np.linspace(-10, 10, 21))

    # p_start = [-8, 9]
    # p_goal = [9, -9]
    # step_len = 0.6
    # random_path_len = 12
    # rrt = RRT(p_start, p_goal, step_len, random_path_len)
    # rrt_path = rrt.random_planning(env_size, poly_list)
    # path_visited = [(rrt.p_start.x, rrt.p_start.y)]
    # print(len(rrt_path))
    # trajectory = rrt.bezier_smooth(rrt_path, 120)
    #
    # ax.scatter(p_start[0], p_start[1], color='b', s=30)
    # # ax.scatter(p_goal[0], p_goal[1], color='r', s=30)
    # plt.ion()
    # for i, node in enumerate(rrt.vertex):
    #     if node.parent:
    #         plt.plot([node.parent.x, node.x], [node.parent.y, node.y], 'limegreen')
    #         plt.gcf().canvas.mpl_connect('key_release_event',
    #                                      lambda event:
    #                                      [exit(0) if event.key == 'escape' else None])
    #         plt.pause(0.001)
    #
    # ax.plot([x[0] for x in rrt_path], [x[1] for x in rrt_path], color='b', linewidth=1.5, marker='o', markersize=2)
    # plt.pause(1)
    # ax.plot([x[0] for x in trajectory], [x[1] for x in trajectory], color='r', linewidth=1.5, marker='o', markersize=2)
    # plt.ioff()
    # plt.show()

    '''
    Multiple Paths
    '''
    poly_list, poly_raw_list = utils.gen_test_env_poly_list_env()
    p_start = [-9.6, 9.6]
    p_start_list = [p_start]
    step_len = 0.5
    random_path_len = 20
    trajectory = []
    new_poly_deque = deque([Point(-10, -10), Point(-10, -10), Point(-10, -10)], maxlen=3)
    poly_list.extend(new_poly_deque)
    while len(trajectory) < 400:
        plt.cla()
        for obs in poly_raw_list:
            x = [obs[num][0] for num in range(len(obs))]
            y = [obs[num][1] for num in range(len(obs))]
            x.append(obs[0][0])
            y.append(obs[0][1])
            ax.plot(x, y, 'k-')
        ax.set_xticks(np.linspace(-10, 10, 21))
        ax.set_yticks(np.linspace(-10, 10, 21))

        rrt = RRT(p_start, (0, 0), step_len, random_path_len)
        rrt_path = rrt.random_planning(env_size, poly_list)
        print(len(rrt_path))
        path_visited = [(rrt.p_start.x, rrt.p_start.y)]
        # print(rrt_path)
        new_traj = bezier_smooth(rrt_path, 400)
        # print(new_traj)
        trajectory = trajectory + new_traj

        ax.scatter([p[0] for p in p_start_list], [p[1] for p in p_start_list], color='b', s=30)
        plt.ion()
        for i, node in enumerate(rrt.vertex):
            if node.parent:
                plt.plot([node.parent.x, node.x], [node.parent.y, node.y], 'limegreen')
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event:
                                             [exit(0) if event.key == 'escape' else None])
                plt.pause(0.001)

        ax.plot([x[0] for x in rrt_path], [x[1] for x in rrt_path], color='b', linewidth=1.5, marker='o', markersize=2)
        plt.pause(1)
        ax.plot([x[0] for x in trajectory], [x[1] for x in trajectory], color='r', linewidth=1.5, marker='o', markersize=2)
        plt.pause(1)

        new_poly_deque.append(LineString(new_traj[:-10]))
        poly_list[-3:] = new_poly_deque
        p_start = trajectory[-1]
        p_start_list.append(p_start)

    plt.cla()
    for obs in poly_raw_list:
        x = [obs[num][0] for num in range(len(obs))]
        y = [obs[num][1] for num in range(len(obs))]
        x.append(obs[0][0])
        y.append(obs[0][1])
        ax.plot(x, y, 'k-')
    ax.set_xticks(np.linspace(-10, 10, 21))
    ax.set_yticks(np.linspace(-10, 10, 21))
    traj = bezier_smooth(trajectory, 1000)
    ax.plot([x[0] for x in traj], [x[1] for x in traj], color='r', linewidth=1.5, marker='o', markersize=2)

    plt.ioff()
    plt.show()