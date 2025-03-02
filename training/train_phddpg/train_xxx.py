import rospy
import time
from torch.utils.tensorboard import SummaryWriter
import os
import sys
sys.path.append('../../')
import training.train_phddpg.agent as agent
from training.environment import GazeboEnvironment
from training.utils import *
from trajectory.trajectory_prediction import KFPredictor


def train_xxx(run_name="hier_DDPG", exp_name="Rand_R1", episode_num=(100, 250, 400, 550),
              iteration_num_start=(200, 400, 500, 600), iteration_num_step=(1, 3, 4, 5),
              iteration_num_max=(1000, 1000, 1000, 1000),
              linear_spd_max=0.6, linear_spd_min=0.05, save_steps=10000,
              env_epsilon=(0.9, 0.6, 0.6), env_epsilon_decay=(0.999, 0.9999, 0.9999),
              laser_half_num=9, goal_th=0.5, obs_th=0.35,
              obs_reward=-20, goal_reward=30, goal_dis_amp=30,
              state_num=20, traj_point_num=150, goal_feature=2, selection_num=1, action_num=2,
              memory_size=100000, batch_size=256, epsilon_end=0.1, rand_start=10000,
              rand_step=2, target_tau=0.01, target_step=1, use_cuda=True):
    '''
    Train XXX (name of HRL algorithm) for motion target mapless navigation

    :param run_name: Name for training run
    :param exp_name: Name for experiment run to get random positions
    :param episode_num: number of episodes for each of the 4 environments
    :param iteration_num_start: start number of maximum steps for 4 environments
    :param iteration_num_step: increase step of maximum steps after each episode
    :param iteration_num_max: max number of maximum steps for 4 environments
    :param linear_spd_max: max wheel speed
    :param linear_spd_min: min wheel speed
    :param save_steps: number of steps to save model
    :param env_epsilon: start epsilon of random action for 4 environments
    :param env_epsilon_decay: decay of epsilon for 4 environments
    :param laser_half_num: half number of scan points
    :param laser_min_dis: min laser scan distance
    :param scan_overall_num: overall number of scan points
    :param goal_dis_min_dis: minimal distance of goal distance
    :param obs_reward: reward for reaching obstacle
    :param goal_reward: reward for reaching goal
    :param goal_dis_amp: amplifier for goal distance change
    :param goal_th: threshold for near a goal
    :param obs_th: threshold for near an obstacle
    :param state_num: number of state
    :param traj_point_num: number of points along the predicted trajectory
    :param goal_feature: feature number of goal
    :param selection_num: number of goal to be selected
    :param action_num: number of action
    :param memory_size: size of memory
    :param batch_size: batch size
    :param epsilon_end: min value for random action
    :param rand_start: max value for random action
    :param rand_decay: steps from max to min random action
    :param rand_step: steps to change
    :param target_tau: update rate for target network
    :param target_step: number of steps between each target update
    :param use_cuda: if true use gpu
    '''
    # Create Folder to save weights
    dirName = 'save_model_weights'
    try:
        os.mkdir('../' + dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

    # Define 3 fixed point training environments
    env1_range, env1_poly_list, env1_raw_poly_list, env1_goal_list, env1_init_list = gen_rand_list_env1(episode_num[0])
    env3_range, env3_poly_list, env3_raw_poly_list, env3_goal_list, env3_init_list = gen_rand_list_env3(episode_num[1])
    env4_range, env4_poly_list, env4_raw_poly_list, env4_goal_list, env4_init_list = gen_rand_list_env4(episode_num[2])
    overall_env_range = [env1_range, env2_range, env3_range, env4_range]
    overall_poly_list = [env1_poly_list, env2_poly_list, env3_poly_list, env4_poly_list]

    overall_init_list = [env1_init_list, env2_init_list, env3_init_list, env4_init_list]
    overall_goal_list = [env1_goal_list, env2_goal_list, env3_goal_list, env4_goal_list]
    # # Read Random Start Pose and Goal Position based on experiment name
    # overall_list = pickle.load(open("../random_positions/" + exp_name + ".p", "rb"))
    # overall_init_list = overall_list[0]
    # overall_goal_list = overall_list[1]
    # print("Use Training Rand Start and Goal Positions: ", exp_name)

    # Define environment
    rospy.init_node("train_xxx")
    env = GazeboEnvironment(laser_scan_half_num=laser_half_num, goal_near_th=goal_th, obs_near_th=obs_th,
                            obs_reward=obs_reward, goal_reward=goal_reward, goal_dis_amp=goal_dis_amp)

    # define trajectory predictor
    pred_tau = 30
    pred_length = 150
    kf_predictor = KFPredictor(pred_tau, pred_length)

    # Define agent object for training
    meta_controller = agent.MetaController(state_num, selection_num, traj_point_num, goal_feature,
                                           memory_size=memory_size, batch_size=batch_size, epsilon_end=epsilon_end,
                                           epsilon_decay_start=rand_start, epsilon_decay=0.9999, epsilon_decay_step=rand_step,
                                           target_tau=target_tau, target_update_steps=target_step, use_cuda=use_cuda)
    controller = agent.Controller(state_num, goal_feature, action_num,
                                  memory_size=memory_size, batch_size=batch_size, epsilon_end=epsilon_end,
                                  epsilon_decay_start=rand_start, epsilon_decay=0.9999, epsilon_decay_step=rand_step,
                                  target_tau=target_tau, target_update_steps=target_step, use_cuda=use_cuda)

    # Define Tensorboard Writer
    tb_writer = SummaryWriter()

    # Define maximum steps per episode and reset maximum random action
    overall_steps = 0
    overall_episode = 0
    env_episode = 0
    env_ita = 0
    ita_per_episode = iteration_num_start[env_ita]
    env.set_new_environment(overall_init_list[env_ita],
                            overall_goal_list[env_ita],
                            overall_poly_list[env_ita])
    meta_controller.reset_epsilon(env_epsilon[env_ita],
                                  env_epsilon_decay[env_ita])
    controller.reset_epsilon(env_epsilon[env_ita],
                             env_epsilon_decay[env_ita])

    # Start Training
    ctrl_steps = 10    # set a constant control step number first, to be modified
    step_penalty = 0.05
    start_time = time.time()
    while True:
        episode_reward = 0
        ita_in_episode = 0
        done = False

        robot_state = env.reset(env_episode)
        scaled_state = state_scale(robot_state)

        observe_start = max(0, ita_in_episode - pred_tau + 1)
        observed_target_traj = env.target_path[observe_start: ita_in_episode + 1]
        pred_target_traj = kf_predictor.predict(observed_target_traj, ita_in_episode)
        encoded_pred_traj = trajectory_encoder(pred_target_traj, robot_state[0])

        while not done and ita_in_episode < ita_per_episode:
            joint_state_traj = np.concatenate([encoded_pred_traj, scaled_state], axis=1)
            raw_goal = meta_controller.select_goal(joint_state_traj)
            goal = encoded_pred_traj[round(raw_goal*(traj_point_num-1))]
            total_ctrl_reward = 0
            step = 0
            while step < ctrl_steps and not done and ita_in_episode < ita_per_episode:
                overall_steps += 1
                ita_in_episode += 1
                step += 1
                joint_state_goal = np.concatenate((goal, scaled_state), axis=0)
                raw_action = controller.act(joint_state_goal)
                action = wheeled_network_2_robot_action(
                    raw_action, linear_spd_max, linear_spd_min
                )
                next_robot_state, reward, done = env.step(action, ita_in_episode)
                # I don't need a goal_reached = xxx, as my goal is always the only moving target
                scaled_next_state = state_scale(next_robot_state)
                next_joint_state_goal = np.concatenate((goal, scaled_next_state), axis=0)
                episode_reward += reward
                total_ctrl_reward += reward
                controller.remember(joint_state_goal, raw_action, reward, next_joint_state_goal, done)
                robot_state = next_robot_state
                scaled_state = scaled_next_state

                # Train network with replay  --to be modified
                if len(controller.memory) > batch_size:
                    actor_loss_value, critic_loss_value = controller.replay()
                    tb_writer.add_scalar('DDPG_Controller/actor_loss', actor_loss_value, overall_steps)
                    tb_writer.add_scalar('DDPG_Controller/critic_loss', critic_loss_value, overall_steps)
                if len(meta_controller.memory) > batch_size:
                    actor_loss_value, critic_loss_value = meta_controller.replay()
                    tb_writer.add_scalar('Meta_Controller/actor_loss', actor_loss_value, overall_steps)
                    tb_writer.add_scalar('Meta_Controller/critic_loss', critic_loss_value, overall_steps)

                # Save Model  --to be modified
                if overall_steps % save_steps == 0:
                    agent.save("../save_model_weights/save_meta_navigator", meta_controller.actor_net, overall_steps // save_steps, run_name)
                    agent.save("../save_model_weights/save_navigator", controller.actor_net, overall_steps // save_steps, run_name)

            observe_start = max(0, ita_in_episode - pred_tau + 1)
            observed_target_traj = env.target_path[observe_start: ita_in_episode + 1]
            next_pred_traj = kf_predictor.predict(observed_target_traj, ita_in_episode)
            encoded_next_traj = trajectory_encoder(next_pred_traj, robot_state[0])
            next_joint_state_traj = np.concatenate([encoded_next_traj, scaled_state], axis=1)
            meta_controller.remember(joint_state_traj, raw_goal, total_ctrl_reward, next_joint_state_traj, done)
            # pred_target_traj = next_pred_traj
            encoded_pred_traj = encoded_next_traj

        print("Episode: {}/{}/{}, Avg Reward: {}, Steps: {}, Env Goal Number: {}"
              .format(env_episode, overall_episode, episode_num, episode_reward / (ita_in_episode + 1),
                      ita_in_episode + 1,
                      len(env.target_init_pos_list)))
        tb_writer.add_scalar('XXX/avg_reward', episode_reward / (ita_in_episode + 1), overall_steps)
        if ita_per_episode < iteration_num_max[env_ita]:
            ita_per_episode += iteration_num_step[env_ita]
        if overall_episode == 999:
            agent.save("../save_model_weights", 0, run_name)
        overall_episode += 1
        env_episode += 1
        if env_episode == episode_num[env_ita]:
            print("Environment ", env_ita, " Training Finished ...")
            if env_ita == 3:
                break
            env_ita += 1
            env.set_new_environment(overall_init_list[env_ita],
                                    overall_goal_list[env_ita],
                                    overall_poly_list[env_ita],
                                    overall_env_range[env_ita])
            meta_controller.reset_epsilon(env_epsilon[env_ita],
                                          env_epsilon_decay[env_ita])
            controller.reset_epsilon(env_epsilon[env_ita],
                                     env_epsilon_decay[env_ita])
            ita_per_episode = iteration_num_start[env_ita]
            env_episode = 0

    end_time = time.time()
    print("Finish Training with time: ", (end_time - start_time) / 60, " Min")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('', type=int, default=0)
    args = parser.parse_args()

    USE_CUDA = True
    if args.cuda == 0:
        USE_CUDA = False

    train_hrl(use_cuda=USE_CUDA)