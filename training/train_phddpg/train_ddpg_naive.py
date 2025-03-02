import rospy
import time
from torch.utils.tensorboard import SummaryWriter
import os
import sys
sys.path.append('../../')
import training.train_phddpg.agent as agent
from training.environment import GazeboEnvironment
from training.utils import *
from evaluation.eval_simulation.utils import *


def train_ddpg_naive(run_name="Naive_DDPG", episode_num=500,
                     iteration_num_start=600, iteration_num_step=5, iteration_num_max=1000,
                     linear_spd_max=0.5, linear_spd_min=0.05, save_steps=10000,
                     env_epsilon=0.1, env_epsilon_decay=0.9999,
                     laser_half_num=9, goal_th=0.5, obs_th=0.35,
                     obs_reward=-20, goal_reward=30, goal_dis_amp=1.5, reward_gamma=0.99,
                     state_num=20, goal_feature=2, action_num=2,
                     epsilon_end=0.01, epsilon_decay_start=10000, epsilon_decay_step=2,
                     memory_size=100000, batch_size=256, target_tau=0.01, target_step=1, use_cuda=True):

    # Create Folder to save weights
    dir_name = 'save_navigator'
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
    rospy.init_node("train_naive_ddpg")
    env = GazeboEnvironment(laser_scan_half_num=laser_half_num, goal_near_th=goal_th, obs_near_th=obs_th,
                            obs_reward=obs_reward, goal_reward=goal_reward, goal_dis_amp=goal_dis_amp)

    # Define agent object for training
    navigator = agent.Controller(state_num, goal_feature, action_num, reward_gamma=reward_gamma,
                                 memory_size=memory_size, batch_size=batch_size, epsilon_end=epsilon_end,
                                 epsilon_decay_start=epsilon_decay_start, epsilon_decay=env_epsilon_decay,
                                 epsilon_decay_step=epsilon_decay_step,
                                 target_tau=target_tau, target_update_steps=target_step, use_cuda=use_cuda)

    actor_net_dir = '../../evaluation/saved_model/ddpg.pt'
    # actor_net_dir = '../save_model_weights/save_navigator/HiDDPG_ctrl_actor_net_999.pt'
    critic_net_dir = '../save_model_weights/save_navigator/HiDDPG_ctrl_critic_net_999.pt'
    navigator.actor_net.load_state_dict(
        torch.load(actor_net_dir, map_location=lambda storage, loc: storage))
    navigator.target_actor_net.load_state_dict(
        torch.load(actor_net_dir, map_location=lambda storage, loc: storage))
    navigator.critic_net.load_state_dict(
        torch.load(critic_net_dir, map_location=lambda storage, loc: storage))
    navigator.target_critic_net.load_state_dict(
        torch.load(critic_net_dir, map_location=lambda storage, loc: storage))

    # Define Tensorboard Writer
    tb_writer = SummaryWriter()

    # Define maximum steps per episode and reset maximum random action
    overall_steps = 0
    overall_episode = 0
    ita_per_episode = iteration_num_start
    env.set_new_environment(robot_init_list,
                            init_target_list,
                            env5_poly_list,
                            env5_range)
    navigator.reset_epsilon(env_epsilon,
                            env_epsilon_decay)

    # Start Training
    start_time = time.time()
    # step_penalty = 0.01
    while True:
        episode_reward = 0
        ita_in_episode = 0
        done = False

        robot_state = env.reset(overall_episode, target_path_list[overall_episode])
        scaled_state = state_scale(robot_state)
        _, encoded_goal = trajectory_encoder(env.target_position, robot_state[0])
        joint_state_goal = np.concatenate((encoded_goal, scaled_state), axis=0)

        while not done and ita_in_episode < ita_per_episode:
            overall_steps += 1
            ita_in_episode += 1

            # action of navigator
            raw_action = navigator.act(joint_state_goal)
            action = wheeled_network_2_robot_action(
                raw_action, linear_spd_max, linear_spd_min
            )
            next_robot_state, _, done, reward, _, success = env.step(action, 'None',
                                                                     [100, 100], [100, 100], [100, 100],
                                                                     ita_in_episode)
            scaled_next_state = state_scale(next_robot_state)
            _, encoded_goal = trajectory_encoder(env.target_position, next_robot_state[0])

            next_joint_state_goal = np.concatenate((encoded_goal, scaled_next_state), axis=0)
            # reward = reward - step_penalty
            episode_reward += reward
            navigator.remember(joint_state_goal, raw_action, reward, next_joint_state_goal, done)
            joint_state_goal = next_joint_state_goal

            # Train network with replay
            if len(navigator.memory) > batch_size:
                actor_loss_value, critic_loss_value = navigator.replay()
                tb_writer.add_scalar('Naive_Navigator/actor_loss', actor_loss_value, overall_steps)
                tb_writer.add_scalar('Naive_Navigator/critic_loss', critic_loss_value, overall_steps)
            tb_writer.add_scalar('Naive_Navigator/action_epsilon', navigator.epsilon, overall_steps)

            # Save Model
            if overall_steps % save_steps == 0:
                agent.save("../save_model_weights/save_navigator",
                           navigator.actor_net, overall_steps // save_steps, run_name)

        print("Episode: {}/{}, Total Reward: {}, Steps: {}, Success: {}"
              .format(overall_episode, episode_num, episode_reward, ita_in_episode + 1, success))
        tb_writer.add_scalar('Naive_Navigator/avg_reward', episode_reward / (ita_in_episode + 1), overall_steps)
        tb_writer.add_scalar('Naive_Navigator/episode_reward', episode_reward, overall_steps)
        if ita_per_episode < iteration_num_max:
            ita_per_episode += iteration_num_step
        if overall_episode == episode_num - 1:
            agent.save("../save_model_weights/save_navigator", navigator.actor_net, 0, run_name)
            print("Naive Navigator Training Finished ...")
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

    train_ddpg_naive(use_cuda=USE_CUDA)
