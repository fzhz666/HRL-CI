from rrt import *
import matplotlib.pyplot as plt
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
fig, ax = plt.subplots(figsize=(10, 10))

poly_list, poly_raw_list = gen_sample_map_poly_list()
env_size = [(-10, 10), (-10, 10)]

p_start = [-4.5, 6.6]
p_start_list = [p_start]
step_len = 0.5
random_path_len = 8
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
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    rrt = RRT(p_start, (0, 0), step_len, random_path_len)
    rrt_path = rrt.random_planning(env_size, poly_list)
    path_visited = [(rrt.p_start.x, rrt.p_start.y)]
    new_traj = bezier_smooth(rrt_path, 200)
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
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
traj = bezier_smooth(trajectory, 1000)
ax.plot([x[0] for x in traj], [x[1] for x in traj], color='r', linewidth=1.5, marker='o', markersize=2)

plt.ioff()
plt.show()