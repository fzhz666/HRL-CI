import rospy
import time
from torch.utils.tensorboard import SummaryWriter
import os
from collections import deque
import sys

sys.path.append('../../')
import training.train_phddpg.agent as agent
from training.environment import GazeboEnvironment
from training.utils import *
from trajectory.trajectory_prediction import KFPredictor
from evaluation.eval_simulation.utils import *

'''

'''

# 消失情况！！！输出预测点，有指数加权+自适应step

def _calculate_trajectory_difference(traj1, traj2):
    """
    Calculate the difference between two trajectories.
    Assumes traj1 and traj2 have the same length.

    :param traj1: First trajectory
    :param traj2: Second trajectory
    :return: Difference between trajectories
    """
    diff = sum(math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) for p1, p2 in zip(traj1, traj2))
    return diff


def update_pred_target_traj(new_pred_target_traj, env_target_path, start_idx):
    average_trajectory_differences = {
        5: 0.20724569078022173,
        6: 0.26518663236276274,
        7: 0.32709062338005335,
        8: 0.39346847553749936,
        9: 0.46478611367059297,
        10: 0.541480560928817,
        11: 0.6239682760854357,
        12: 0.712649841009502,
        13: 0.8079168199875919,
        14: 0.9101548275930158,
        15: 1.0197479804143088
    }
    # pred_target_traj_count是上一次循环结束的预测路径总个数
    # self.all_pred_target_traj.append(new_pred_target_traj)

    # 计算当前导航的 trajectory differences 并存储
    # num_points_to_compare = [5, 10, 15]
    num_points_to_compare = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # 从5到15的时间步选择
    selected_time_step = 4

    found_time_step = False  # 初始化标志变量

    for num_points in num_points_to_compare:
        # start_idx = (len(self.all_pred_target_traj) - pred_target_traj_count - 1) * ctrl_steps
        current_traj = env_target_path[start_idx: start_idx + num_points - 1]  # 带第一次数据
        predicted_traj = new_pred_target_traj[:num_points - 1]
        diff = _calculate_trajectory_difference(current_traj, predicted_traj)
        print('diff = ', diff)

        # 从5到15的时间步选择
        average_diff = average_trajectory_differences[num_points]
        # print('num_points = ', num_points)
        if diff >= average_diff:  # 修改条件为大于等于
            selected_time_step = num_points
            found_time_step = True
            # print('num_points = ', num_points)
            break

    # 如果所有的diff都小于对应的average_diff，则selected_time_step等于15
    if not found_time_step:
        selected_time_step = 15

    return selected_time_step

def train_meta_controller(run_name="Train_Meta_index_00", episode_num=500,
                          iteration_num_start=600, iteration_num_step=5, iteration_num_max=1000,
                          linear_spd_max=0.5, linear_spd_min=0.05, save_steps=10000,
                          env_epsilon=1, env_epsilon_decay=0.9997,
                          laser_half_num=9, goal_th=0.5, obs_th=0.35,
                          obs_reward=-20, goal_reward=30, goal_dis_amp=30, reward_gamma=1,
                          state_num=20, traj_point_num=150, goal_feature=2, selection_num=1,
                          epsilon_end=0.1, epsilon_decay_start=1000, epsilon_decay_step=2,
                          memory_size=10000, batch_size=256, target_tau=0.01, target_step=1, use_cuda=True):
    # Create Folder to save weights
    dir_name = 'compare_queue5_a0.5'  # 尝试过5了
    try:
        os.makedirs('../save_model_weights/' + dir_name)
        print("Directory ", dir_name, " Created ")
    except FileExistsError:
        print("Directory ", dir_name, " already exists")

    # Define training environment
    env5_range, env5_poly_list, env5_raw_poly_list = gen_poly_list_env5()
    rand_paths_robot_list = pickle.load(open('train_paths_robot_pose.p', 'rb'))
    robot_init_list = rand_paths_robot_list[0][:]
    target_path_list = rand_paths_robot_list[1][:]
    init_target_list = [path[0] for path in target_path_list]

    # Define environment
    rospy.init_node("train_meta_controller")
    env = GazeboEnvironment(laser_scan_half_num=laser_half_num, goal_near_th=goal_th, obs_near_th=obs_th,
                            obs_reward=obs_reward, goal_reward=goal_reward, goal_dis_amp=goal_dis_amp)

    # define trajectory predictor
    pred_tau = 30
    pred_length = 150
    kf_predictor = KFPredictor(pred_tau, pred_length)

    # Define agent object for training
    meta_controller = agent.MetaController(state_num, selection_num, traj_point_num, goal_feature,
                                           reward_gamma=reward_gamma, memory_size=memory_size, batch_size=batch_size,
                                           epsilon_end=epsilon_end,
                                           epsilon_decay_start=epsilon_decay_start, epsilon_decay=env_epsilon_decay,
                                           epsilon_decay_step=epsilon_decay_step,
                                           target_tau=target_tau, target_update_steps=target_step, use_cuda=use_cuda)
    net_dir = '../../evaluation/saved_model/ddpg.pt'
    # net_dir = '../save_model_weights/save_navigator/HiDDPG_ctrl_actor_net_999.pt'
    controller_net = load_actor_net(net_dir)
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    controller_net.to(device)

    # Define Tensorboard Writer
    tb_writer = SummaryWriter()

    # Define maximum steps per episode and reset maximum random action
    overall_steps = 0
    mate_overall_steps = 0
    overall_episode = 0
    ita_per_episode = iteration_num_start
    env.set_new_environment(robot_init_list,
                            init_target_list,
                            env5_poly_list,
                            env5_range)
    meta_controller.reset_epsilon(env_epsilon,
                                  env_epsilon_decay)

    # Start Training
    ctrl_steps = 10  # set a constant control step number first, to be modified
    step_penalty = 0.05  # hyper-parameter, maybe need adjusting
    start_time = time.time()

    ###
    # 在训练循环开始之前初始化一个队列，用于存储预测目标轨迹
    deque_number = 5  # 设置有几个队列
    target_traj_queue = deque(maxlen=deque_number)  # 最多存储deque_number条轨迹
    # 计算指数加权系数，可以根据需要调整
    a = 0.5  # 由你自己设置，取值范围在0到1之间。a越小越看重最新的路径权重

    while True:
        target_disappear_steps = 0
        target_disappear_duration = 3  # 目标消失的步数
        target_disappear_frequency = 10  # target_disappear_frequency次step循环内套target_disappear_duration次消失
        disappear_flag = 0
        observe_start_disappear = 0
        start_idx = 0

        episode_meta_reward = 0
        ita_in_episode = 0
        shuzi = 0  # 用来作为队列初始化的计算
        done = False
        disappear_num = 0  # 计算消失步

        robot_state = env.reset(overall_episode, target_path_list[overall_episode])
        scaled_state = state_scale(robot_state)

        observe_start = max(0, ita_in_episode - pred_tau + 1)
        # print('ita_in_episode+1 =', ita_in_episode + 1)
        observed_target_traj = env.target_path[observe_start: ita_in_episode + 1]
        pred_target_traj = kf_predictor.predict(observed_target_traj, ita_in_episode)

        ###
        while shuzi != deque_number:
            target_traj_queue.append(pred_target_traj)  # 目标预测路径队列初始化，全部设置为第一条
            shuzi += 1

        # 初始化综合的预测路径
        combined_pred_traj = np.zeros_like(pred_target_traj)
        combined_pred_traj += (a**(deque_number-1)) * target_traj_queue[0]  # 第一条预测路径乘以a
        weight = 1 - a
        # print('len(target_traj_queue) = ', len(target_traj_queue))
        for i in range(1, len(target_traj_queue)):
            combined_pred_traj += weight * target_traj_queue[len(target_traj_queue)-i]  # 后续路径乘以(1-a)的累计系数
            weight *= a  # 更新权重，叠加(1-a)的次方
        # 手动调整综合轨迹的起点为目标点的起点
        offset = pred_target_traj[0] - combined_pred_traj[0]
        combined_pred_traj += offset
        pred_target_traj = combined_pred_traj

        # 模拟多次更新预测轨迹
        new_pred_target_traj = pred_target_traj  # 请替换成实际的新预测轨迹
        env_target_path = env.target_path  # 请替换成实际的环境目标轨迹
        ctrl_steps = update_pred_target_traj(new_pred_target_traj, env_target_path, start_idx)
        start_idx += ctrl_steps

        # Attention: add flatting operation in trajectory_encoder(), see whether it works
        encoded_pred_traj, flatted_pred_traj = trajectory_encoder(pred_target_traj, robot_state[0])

        while not done and ita_in_episode < ita_per_episode:
            # print('跑一次10步')
            joint_state_traj = np.concatenate([flatted_pred_traj, scaled_state], axis=0)
            raw_goal = meta_controller.select_goal(joint_state_traj)

            goal_option = int(np.round(raw_goal * (traj_point_num - 1)))
            # print('goal_option= ', goal_option)
            # if target_disappear_steps > 0:
            #     print('处于消失过程中')
            #     print('goal_option = ', goal_option)

            goal = encoded_pred_traj[goal_option]
            step = 0

            mate_overall_steps += 1
            disappear_num += 1  # 计算消失步

            while step < ctrl_steps and not done and ita_in_episode < ita_per_episode:
                # if disappear_flag == 1:
                #     observe_start_disappear += 1
                if disappear_flag == 2:
                    observe_start_disappear += 1  # 从消失到显现后开始计算的步数
                overall_steps += 1
                ita_in_episode += 1
                step += 1
                # if target_disappear_steps > 0:
                #     print('处于消失过程中')
                #     print('scaled_state = ', scaled_state)
                joint_state_goal = np.concatenate((goal, scaled_state), axis=0)

                # action of controller
                state = np.array(joint_state_goal).reshape((1, -1))
                state = torch.Tensor(state).to(device)
                action_tmp = controller_net(state).to('cpu')
                action_tmp = action_tmp.detach().numpy().squeeze()
                noise = np.random.randn(2) * 0.01
                action_tmp = noise + (1 - 0.01) * action_tmp
                raw_action = np.clip(action_tmp, [0., 0.], [1., 1.])

                action = wheeled_network_2_robot_action(
                    raw_action, linear_spd_max, linear_spd_min
                )
                # next_robot_state, extrinsic_reward, done, _, _, _ = env.step(action, pred_target_traj[goal_option],
                #                                                              pred_target_traj[49],
                #                                                              pred_target_traj[99],
                #                                                              pred_target_traj[149], ita_in_episode)
                next_robot_state, extrinsic_reward, done, _, _, _ = env.step1(action, pred_target_traj[goal_option],
                                                                              pred_target_traj[0],
                                                                              pred_target_traj[49],
                                                                              pred_target_traj[99],
                                                                              pred_target_traj[149], ita_in_episode)
                scaled_next_state = state_scale(next_robot_state)
                # next_joint_state_goal = np.concatenate((goal, scaled_next_state), axis=0)
                # episode_reward += rewarddcua
                robot_state = next_robot_state
                scaled_state = scaled_next_state

            if disappear_num % target_disappear_frequency == 0:
                target_disappear_steps = target_disappear_duration + 1
                disappear_flag = 1
                observe_start_disappear = 0

            if target_disappear_steps > 0:
                target_disappear_steps -= 1

            if target_disappear_steps == 0:
                observe_start = max(0, ita_in_episode - pred_tau + 1)
                observed_traj_num = ita_in_episode

                # print('observe_start =', observe_start)
                if observe_start_disappear < pred_tau and disappear_flag == 2:
                    # print('进入flag=2')
                    observe_start = ita_in_episode - observe_start_disappear
                    observed_traj_num = observe_start_disappear

                if observe_start_disappear >= pred_tau:
                    disappear_flag = 0
                if disappear_flag == 1:
                    observe_start_disappear = 0
                    observe_start = ita_in_episode
                    observed_traj_num = observe_start_disappear
                    # print('进入flag=1')
                    disappear_flag = 2
                # print('observe_start1 =', observe_start)
                # print('observed_traj_num+1 =', observed_traj_num + 1)
                # print('........')
                observed_target_traj = env.target_path[observe_start: ita_in_episode + 1]
                # observed_traj_num = ita_in_episode
                next_pred_traj = kf_predictor.predict(observed_target_traj, observed_traj_num)

                ###
                target_traj_queue.append(next_pred_traj)
                combined_pred_traj = np.zeros_like(pred_target_traj)
                combined_pred_traj += (a**(deque_number-1)) * target_traj_queue[0]  # 第一条预测路径乘以a
                weight = 1 - a  # a越小越看重最新的路径权重
                # print('len(target_traj_queue) = ', len(target_traj_queue))
                for i in range(1, len(target_traj_queue)):
                    combined_pred_traj += weight * target_traj_queue[len(target_traj_queue)-i]  # 后续路径乘以(1-a)的累计系数
                    weight *= a  # 更新权重，叠加(1-a)的次方
                # next_pred_traj = combined_pred_traj
                # 手动调整综合轨迹的起点为目标点的起点
                offset = next_pred_traj[0] - combined_pred_traj[0]  # 这里要用next_pred_traj作为加减
                combined_pred_traj += offset
                next_pred_traj = combined_pred_traj

            encoded_next_traj, flatted_next_traj = trajectory_encoder(next_pred_traj, robot_state[0])
            next_joint_state_traj = np.concatenate([flatted_next_traj, scaled_state], axis=0)
            meta_reward = extrinsic_reward - step_penalty * step
            episode_meta_reward += meta_reward
            meta_controller.remember(joint_state_traj, raw_goal, meta_reward, next_joint_state_traj, done)
            pred_target_traj = next_pred_traj
            encoded_pred_traj = encoded_next_traj
            flatted_pred_traj = flatted_next_traj

            # Train network with replay
            if len(meta_controller.memory) > batch_size:
                actor_loss_value, critic_loss_value = meta_controller.replay()
                tb_writer.add_scalar('Meta_Controller/actor_loss', actor_loss_value, overall_steps)
                tb_writer.add_scalar('Meta_Controller/critic_loss', critic_loss_value, overall_steps)
            tb_writer.add_scalar('Meta_Controller/action_epsilon', meta_controller.epsilon, overall_steps)

            # Save Model
            if mate_overall_steps % (save_steps / 4) == 0:  # 20次一个模型(/10)
                agent.save("../save_model_weights/" + dir_name, meta_controller.actor_net,
                           mate_overall_steps // (save_steps / 4), run_name)

        print("Episode: {}/{}, Total Reward: {:.5f}, Avg Reward: {:.8f}, Steps: {}"
              .format(overall_episode, episode_num, episode_meta_reward, episode_meta_reward / (ita_in_episode + 1),
                      ita_in_episode + 1))
        tb_writer.add_scalar('XXX/avg_reward', episode_meta_reward / (ita_in_episode + 1), overall_steps)
        tb_writer.add_scalar('XXX/episode_reward', episode_meta_reward, overall_steps)
        if ita_per_episode < iteration_num_max:
            ita_per_episode += iteration_num_step
        if overall_episode == episode_num - 1:
            agent.save("../save_model_weights" + dir_name, meta_controller.actor_net, 0, run_name)
            print("Meta Controller Training Finished ...")
            break
        overall_episode += 1

    end_time = time.time()
    print("Finish Training with time: ", (end_time - start_time) / 60, " Min")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1)
    args = parser.parse_args()

    USE_CUDA = True
    if args.cuda == 0:
        USE_CUDA = False

    train_meta_controller(use_cuda=USE_CUDA)
