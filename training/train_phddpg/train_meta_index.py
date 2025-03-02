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
1. ctrl_steps = 10, step_penalty = 1, reward_gamma = 0.99, dist_reward = None, meta_controller train step = env-step
   epsilon = 0.6, epsilon_end = 0.1, epsilon_decay = 0.9999, epsilon_decay_start = 10000, epsilon_decay_step = 2
   noise: same way as controller —— noise = np.random.randn(self.action_num) * self.epsilon
                                    selection = noise + (1 - self.epsilon) * selection
   action_rand(eval) = 0.01
   result: 53% success rate, long avg-dist & slow avg-spd

2. ctrl_steps = 10, step_penalty = 0.05, reward_gamma = 1, dist_reward = None, meta_controller train step = ctrl_steps * env-step
   epsilon = 1, epsilon_end = 0.1, epsilon_decay = 0.9997, epsilon_decay_start = 1000, epsilon_decay_step = 2
   noise: noise = np.random.normal(0, 0.5) * self.epsilon
          selection = noise + selection
   noise(eval) = 0
   result: Success = 161  Collision = 16  Overtime = 23, avg-dist = 17.4 m, avg-time = 42.9 s, avg-spd = 0.4 m/s, but the actor loss and critic loss stayed 0 throughout the training
   (random-trained with ctrl-steps = 5 & r1-eval: 87%, 18.7m, 43.9s, 0.41m/s)
   meta-controller actor net: s22~s27

2-plus: ctrl_steps = 10, step_penalty = 0.05, reward_gamma = 1, dist_reward = None, meta_controller train step = ctrl_steps * env-step
   epsilon = 1, epsilon_end = 0.1, epsilon_decay = 0.9997, epsilon_decay_start = 1000, epsilon_decay_step = 2
   noise: noise = np.random.normal(0, 0.5) * self.epsilon
          selection = noise + selection
   noise(eval) = 0
   modified KF (Q is modified to the strict form)
   memory_size = 10000, training episode = 500
   (memory size for meta-controller has been changed into 10000, the previous is too large because of ignoring ctrl_steps or training step(10))
   result: Success = 160  Collision = 20  Overtime = 20, avg-dist = 16.93689 m, avg-time = 41.518 s, avg-spd = 0.40646 m/s

3. ctrl_steps = 10, step_penalty = 0.02, reward_gamma = 1, dist_reward = None, meta_controller train step = ctrl_steps * env-step
   epsilon = 1, epsilon_end = 0.05, epsilon_decay = 0.9997, epsilon_decay_start = 1000, epsilon_decay_step = 2
   noise: noise = np.random.normal(0, 0.5) * self.epsilon
          selection = noise + selection
   noise(eval) = 0
   result: Success = 155  Collision = 12  Overtime = 33, avg-dist = 18.8 m, avg-time = 44.8 s, avg-spd = 0.41 m/s
   meta-controller actor net: s0~s21

5. lower-level param: env_episode = (100, 250, 400), spd_range = (0.05, 0.6), epsilon = (0.9, 0.6, 0.6), epsilon_decay = (0.999, 0.9999, 0.9999)
   epsilon_end = 0.05, epsilon_decay_start = 10000, noise = N(0, 0.25), 
   NO, this param is not stable and cannot work well. especially note this: "action = action * (1 - epsilon) + noise * epsilon" produce smoother actions 
   than "action = action + noise * epsilon" 
'''

# 输出预测点，有指数加权
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
    dir_name = 'meta_index_3_0.5_step_5'  # 尝试过5了
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
    ctrl_steps = 5  # set a constant control step number first, to be modified
    step_penalty = 0.05  # hyper-parameter, maybe need adjusting
    start_time = time.time()

    ###
    # 在训练循环开始之前初始化一个队列，用于存储预测目标轨迹
    deque_number = 3  # 设置有几个队列
    target_traj_queue = deque(maxlen=deque_number)  # 最多存储deque_number条轨迹
    # 计算指数加权系数，可以根据需要调整
    a = 0.5  # 由你自己设置，取值范围在0到1之间
    # shuzi = 0  # 用来作为队列初始化的计算

    while True:
        episode_meta_reward = 0
        ita_in_episode = 0
        shuzi = 0  # 用来作为队列初始化的计算
        done = False

        robot_state = env.reset(overall_episode, target_path_list[overall_episode])
        scaled_state = state_scale(robot_state)

        observe_start = max(0, ita_in_episode - pred_tau + 1)
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

        # Attention: add flatting operation in trajectory_encoder(), see whether it works
        encoded_pred_traj, flatted_pred_traj = trajectory_encoder(pred_target_traj, robot_state[0])

        while not done and ita_in_episode < ita_per_episode:
            joint_state_traj = np.concatenate([flatted_pred_traj, scaled_state], axis=0)
            raw_goal = meta_controller.select_goal(joint_state_traj)

            # raw_goal, ctrl_steps = meta_controller.select_goal_step(joint_state_traj)
            # print('raw_goal = ', raw_goal)  # 输出一个数

            goal_option = int(np.round(raw_goal * (traj_point_num - 1)))
            # print('goal_option= ', goal_option)

            # ctrl_steps = ctrl_steps * 10 + 5  # ctrl_steps范围在5~15
            # print('ctrl_steps = ', ctrl_steps)

            goal = encoded_pred_traj[goal_option]
            step = 0

            mate_overall_steps += 1
            while step < ctrl_steps and not done and ita_in_episode < ita_per_episode:
                overall_steps += 1
                ita_in_episode += 1
                step += 1
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
                next_robot_state, extrinsic_reward, done, _, _, _ = env.step(action, pred_target_traj[goal_option],
                                                                             pred_target_traj[49],
                                                                             pred_target_traj[99],
                                                                             pred_target_traj[149], ita_in_episode)
                scaled_next_state = state_scale(next_robot_state)
                # next_joint_state_goal = np.concatenate((goal, scaled_next_state), axis=0)
                # episode_reward += reward
                robot_state = next_robot_state
                scaled_state = scaled_next_state

            observe_start = max(0, ita_in_episode - pred_tau + 1)
            observed_target_traj = env.target_path[observe_start: ita_in_episode + 1]
            next_pred_traj = kf_predictor.predict(observed_target_traj, ita_in_episode)

            ###
            target_traj_queue.append(next_pred_traj)
            combined_pred_traj = np.zeros_like(pred_target_traj)
            combined_pred_traj += (a**(deque_number-1)) * target_traj_queue[0]  # 第一条预测路径乘以a
            weight = 1 - a
            # print('len(target_traj_queue) = ', len(target_traj_queue))
            for i in range(1, len(target_traj_queue)):
                combined_pred_traj += weight * target_traj_queue[len(target_traj_queue)-i]  # 后续路径乘以(1-a)的累计系数
                weight *= a  # 更新权重，叠加(1-a)的次方
                # 手动调整综合轨迹的起点为目标点的起点
            offset = next_pred_traj[0] - combined_pred_traj[0]  # 这里要用next_pred_traj作为加减，在index文件中没有用到！
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
