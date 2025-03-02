import rospy
import time
from torch.utils.tensorboard import SummaryWriter
import pickle
import os
import sys
sys.path.append('../../')
import training.train_ddpg.agent as agent
from training.environment import GazeboEnvironment
from training.utils import *
from trajectory.trajectory_prediction import KFPredictor


'''
env2 needs at least 150 episodes
from the training performance and actor loss curve we can see, things become different when the epsilon reached 0.1. 
The continual decay to 0.05 will leads to bad performance and if it stays at 0.1, performance will keep getting better.
robot 原地转圈，因为上层给出的目标点一直在变，且真实目标一直在运动，这使得下层的 intrinsic reward 可以非常大。
即便目标送到嘴边了也不去达到，最后走了691步，还能获得299.42的奖励。试试不用距离奖励，只用步数惩罚？
'''


def staged_train_hrl(run_name="HiDDPG", episode_num=(100, 200, 300, 400, 500),
                     iteration_num_start=(200, 300, 400, 500, 600), iteration_num_step=(1, 2, 3, 4, 5),
                     iteration_num_max=(1000, 1000, 1000, 1000, 1000),
                     env_epsilon=((0, 0.9), (0, 0.6), (0, 0.6), (0, 0.6), (1, 0.1)),
                     env_epsilon_decay=((0, 0.999), (0, 0.9999), (0, 0.9999), (0, 0.9999), (0.9997, 0.99995)),
                     epsilon_end=(0.1, 0.1), epsilon_decay_start=(1000, 10000), epsilon_decay_step=2,
                     target_tau=0.01, target_step=1,
                     linear_spd_max=0.5, linear_spd_min=0.05, laser_half_num=9, goal_th=0.5, obs_th=0.35,
                     obs_reward=-20, goal_reward=30, goal_dis_amp=15,
                     state_num=20, traj_point_num=150, goal_feature=2, selection_num=1, action_num=2,
                     memory_size=(10000, 100000), batch_size=256, save_steps=10000,
                     use_cuda=True):
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
    :param epsilon_decay_start: max value for random action
    :param rand_decay: steps from max to min random action
    :param epsilon_decay_step: steps to change
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

    # Define training environments
    env1_range, env1_poly_list, env1_raw_poly_list, env1_goal_list, env1_init_list = gen_rand_list_env1(episode_num[0])
    env2_range, env2_poly_list, env2_raw_poly_list, env2_goal_list, env2_init_list = gen_rand_list_env2(episode_num[1])
    env3_range, env3_poly_list, env3_raw_poly_list, env3_goal_list, env3_init_list = gen_rand_list_env3(episode_num[2])
    env4_range, env4_poly_list, env4_raw_poly_list, env4_goal_list, env4_init_list = gen_rand_list_env4(episode_num[3])
    env5_range, env5_poly_list, env5_raw_poly_list = gen_poly_list_env5()

    # Read Random Start Pose and Goal Position based on experiment name
    # rand_option = "Rand_R1"
    # overall_list = pickle.load(open("../random_positions/" + rand_option + ".p", "rb"))
    # overall_init_list = overall_list[0]
    # overall_goal_list = overall_list[1]
    # print("Use Training Rand Start and Goal Positions in First 4 Environments: ", rand_option)

    rand_paths_robot_list = pickle.load(open('train_paths_robot_pose.p', 'rb'))
    env5_init_list = rand_paths_robot_list[0][:]
    target_path_list = rand_paths_robot_list[1][:]
    env5_goal_list = [path[0] for path in target_path_list]

    overall_env_range = [env1_range, env2_range, env3_range, env4_range, env5_range]
    overall_poly_list = [env1_poly_list, env2_poly_list, env3_poly_list, env4_poly_list, env5_poly_list]
    # overall_init_list.append(env5_init_list)
    # overall_goal_list.append(env5_goal_list)
    overall_init_list = [env1_init_list, env2_init_list, env3_init_list, env4_init_list, env5_init_list]
    overall_goal_list = [env1_goal_list, env2_goal_list, env3_goal_list, env4_goal_list, env5_goal_list]

    # Define environment
    rospy.init_node("train_xxx_in_stages")
    env = GazeboEnvironment(laser_scan_half_num=laser_half_num, goal_near_th=goal_th, obs_near_th=obs_th,
                            obs_reward=obs_reward, goal_reward=goal_reward, goal_dis_amp=goal_dis_amp)

    # define trajectory predictor
    pred_tau = 30
    pred_length = 150
    kf_predictor = KFPredictor(pred_tau, pred_length)

    # Define agent object for training
    meta_controller = agent.MetaController(state_num, selection_num, traj_point_num, goal_feature,
                                           memory_size=memory_size[0], batch_size=batch_size,
                                           epsilon_end=epsilon_end[0], epsilon_decay_start=epsilon_decay_start[0],
                                           epsilon_decay_step=epsilon_decay_step,
                                           target_tau=target_tau, target_update_steps=target_step, use_cuda=use_cuda)
    controller = agent.Controller(state_num, goal_feature, action_num,
                                  memory_size=memory_size[1], batch_size=batch_size,
                                  epsilon_end=epsilon_end[1], epsilon_decay_start=epsilon_decay_start[1],
                                  epsilon_decay_step=epsilon_decay_step,
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
                            overall_poly_list[env_ita],
                            overall_env_range[env_ita])
    meta_controller.reset_epsilon(env_epsilon[env_ita][0],
                                  env_epsilon_decay[env_ita][0])
    controller.reset_epsilon(env_epsilon[env_ita][1],
                             env_epsilon_decay[env_ita][1])

    # Start Training
    concurrent_train = False
    ctrl_steps = 10
    meta_step_penalty = 0.05
    start_time = time.time()
    while True:
        episode_meta_reward = 0
        episode_intrinsic_reward = 0
        ita_in_episode = 0
        done = False

        if env_ita < 4:
            robot_state = env.reset(env_episode, new_target_path=None)
            scaled_state = state_scale(robot_state)
        elif env_ita == 4:
            robot_state = env.reset(env_episode, target_path_list[env_episode])
            scaled_state = state_scale(robot_state)
            observe_start = max(0, ita_in_episode - pred_tau + 1)
            observed_target_traj = env.target_path[observe_start: ita_in_episode + 1]
            pred_target_traj = kf_predictor.predict(observed_target_traj, ita_in_episode)
            encoded_pred_traj, flatted_pred_traj = trajectory_encoder(pred_target_traj, robot_state[0])
        else:
            print("Wrong 'env_ita' value!")

        while not done and ita_in_episode < ita_per_episode:
            intrinsic_done = False
            if env_ita < 4:
                goal_dis, goal_dir = robot_2_goal_dis_dir(env.target_position, robot_state[0])
                scaled_dis = goal_dis if goal_dis != 0 else 0.3
                scaled_dis = 0.3 / scaled_dis
                scaled_dis = scaled_dis if scaled_dis <= 1 else 1
                scaled_dir = goal_dir / math.pi
                goal = np.array([scaled_dir, scaled_dis])
            elif env_ita == 4:
                joint_state_traj = np.concatenate([flatted_pred_traj, scaled_state], axis=0)
                raw_goal = meta_controller.select_goal(joint_state_traj)
                goal_option = int(np.round(raw_goal * (traj_point_num - 1)))
                goal = encoded_pred_traj[goal_option]
            else:
                print("Wrong 'env_ita' value!")
            step = 0
            while step < ctrl_steps and not intrinsic_done and not done and ita_in_episode < ita_per_episode:
                overall_steps += 1
                ita_in_episode += 1
                step += 1
                joint_state_goal = np.concatenate((goal, scaled_state), axis=0)
                raw_action = controller.act(joint_state_goal)
                action = wheeled_network_2_robot_action(
                    raw_action, linear_spd_max, linear_spd_min
                )
                if env_ita < 4:
                    next_robot_state, extrinsic_reward, done, \
                      intrinsic_reward, intrinsic_done = env.step(action, env.target_position,
                                                                  [-100, -100], [-100, -100],
                                                                  [-100, -100], ita_in_episode)
                elif env_ita == 4:
                    next_robot_state, extrinsic_reward, done, \
                      intrinsic_reward, intrinsic_done = env.step(action, pred_target_traj[goal_option],
                                                                  pred_target_traj[49],
                                                                  pred_target_traj[99],
                                                                  pred_target_traj[149], ita_in_episode)
                else:
                    print("Wrong 'env_ita' value!")
                scaled_next_state = state_scale(next_robot_state)
                next_joint_state_goal = np.concatenate((goal, scaled_next_state), axis=0)

                episode_intrinsic_reward += intrinsic_reward
                controller.remember(joint_state_goal, raw_action, intrinsic_reward, next_joint_state_goal, intrinsic_done)
                robot_state = next_robot_state
                scaled_state = scaled_next_state

                # Train network with replay
                if (len(controller.memory) > batch_size and env_ita < 4) or concurrent_train:
                    actor_loss_value, critic_loss_value = controller.replay()
                    tb_writer.add_scalar('Controller/actor_loss', actor_loss_value, overall_steps)
                    tb_writer.add_scalar('Controller/critic_loss', critic_loss_value, overall_steps)
                tb_writer.add_scalar('Controller/action_epsilon', controller.epsilon, overall_steps)
                if env_ita == 4 and env_episode == 100:
                    controller.reset_epsilon(0.3, env_epsilon_decay[env_ita][1])
                    concurrent_train = True

                # Save Model
                if overall_steps % save_steps == 0:
                    agent.save("../save_model_weights/save_navigator", controller.actor_net, overall_steps // save_steps, run_name + '_ctrl')

            if env_ita == 4:
                observe_start = max(0, ita_in_episode - pred_tau + 1)
                observed_target_traj = env.target_path[observe_start: ita_in_episode + 1]
                next_pred_traj = kf_predictor.predict(observed_target_traj, ita_in_episode)
                encoded_next_traj, flatted_next_traj = trajectory_encoder(next_pred_traj, robot_state[0])
                next_joint_state_traj = np.concatenate([flatted_next_traj, scaled_state], axis=0)
                meta_reward = extrinsic_reward - meta_step_penalty * step
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
                if overall_steps % save_steps == 0:
                    agent.save("../save_model_weights/save_meta_navigator", meta_controller.actor_net,
                               overall_steps // save_steps, run_name + '_meta_ctrl')

        print("Episode: {}/{}/{}, Meta-Controller Reward: {:.2f}, Controller Reward: {:.2f}, Steps: {}"
              .format(env_episode, overall_episode, episode_num, episode_meta_reward,
                      episode_intrinsic_reward, ita_in_episode + 1))
        tb_writer.add_scalar('Meta_Controller/reward', episode_meta_reward, overall_steps)
        tb_writer.add_scalar('Controller/reward', episode_intrinsic_reward, overall_steps)

        if ita_per_episode < iteration_num_max[env_ita]:
            ita_per_episode += iteration_num_step[env_ita]
        # for testing simultaneous learning, save controller's critic
        if overall_episode == 599:
            agent.save("../save_model_weights/save_navigator", controller.critic_net, overall_steps // save_steps, run_name + '_critic_of')
        if overall_episode == 999:
            agent.save("../save_model_weights/save_navigator", controller.critic_net, overall_steps // save_steps, run_name + '_critic_of')
        if overall_episode == 1499:
            agent.save("../save_model_weights/save_meta_navigator", meta_controller.actor_net, 0, run_name + '_meta_ctrl')
            print("Meta Controller Training Finished ...")
            agent.save("../save_model_weights/save_navigator", controller.actor_net, 0, run_name + '_ctrl')
            print("Controller Training Finished ...")
        overall_episode += 1
        env_episode += 1
        if env_episode == episode_num[env_ita]:
            print("Environment ", env_ita, " Training Finished ...")
            if env_ita == 4:
                break
            env_ita += 1
            env.set_new_environment(overall_init_list[env_ita],
                                    overall_goal_list[env_ita],
                                    overall_poly_list[env_ita],
                                    overall_env_range[env_ita])
            meta_controller.reset_epsilon(env_epsilon[env_ita][0],
                                          env_epsilon_decay[env_ita][0])
            controller.reset_epsilon(env_epsilon[env_ita][1],
                                     env_epsilon_decay[env_ita][1])
            ita_per_episode = iteration_num_start[env_ita]
            env_episode = 0

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

    staged_train_hrl(use_cuda=USE_CUDA)
