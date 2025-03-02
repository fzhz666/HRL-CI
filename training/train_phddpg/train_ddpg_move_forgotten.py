import rospy
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import pickle
import os
import sys
sys.path.append('../../')
import training.train_phddpg.agent as agent
from training.environment import GazeboEnvironment
from training.utils import *
from trajectory.trajectory_prediction import KFPredictor


def train_ddpg_move(run_name="DDPG", episode_num=(100, 200, 300, 400, 500),
                    iteration_num_start=(200, 300, 400, 500, 600), iteration_num_step=(1, 2, 3, 4, 5),
                    iteration_num_max=(1000, 1000, 1000, 1000, 1000),
                    env_epsilon=((0, 0.9), (0, 0.6), (0, 0.6), (0, 0.6), (1, 0.2)),
                    env_epsilon_decay=((0, 0.999), (0, 0.9999), (0, 0.9999), (0, 0.9999), (0.9997, 0.99995)),
                    epsilon_end=(0.1, 0.1), epsilon_decay_start=(1000, 10000), epsilon_decay_step=2,
                    target_tau=0.01, target_step=1,
                    linear_spd_max=0.5, linear_spd_min=0.05, laser_half_num=9, goal_th=0.5, obs_th=0.35,
                    obs_reward=-20, goal_reward=30, goal_dis_amp=15,
                    state_num=20, traj_point_num=150, goal_feature=2, selection_num=1, action_num=2,
                    memory_size=(10000, 100000), batch_size=256, save_steps=10000,
                    use_cuda=True):
    # Create Folder to save weights
    dirName = 'save_model_weights'
    try:
        os.mkdir('../' + dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

    # Define 3 fixed point training environments
    env1_range, env1_poly_list, env1_raw_poly_list, env1_goal_list, env1_init_list = gen_rand_list_env1(episode_num[0])
    env2_range, env2_poly_list, env2_raw_poly_list, env2_goal_list, env2_init_list = gen_rand_list_env2(episode_num[1])
    env3_range, env3_poly_list, env3_raw_poly_list, env3_goal_list, env3_init_list = gen_rand_list_env3(episode_num[2])
    env4_range, env4_poly_list, env4_raw_poly_list, env4_goal_list, env4_init_list = gen_rand_list_env4(episode_num[3])
    env5_range, env5_poly_list, env5_raw_poly_list = gen_poly_list_env5()

    rand_paths_robot_list = pickle.load(open('train_paths_robot_pose.p', 'rb'))
    env5_init_list = rand_paths_robot_list[0][:]
    target_path_list = rand_paths_robot_list[1][:]
    env5_goal_list = [path[0] for path in target_path_list]

    overall_env_range = [env1_range, env2_range, env3_range, env4_range, env5_range]
    overall_poly_list = [env1_poly_list, env2_poly_list, env3_poly_list, env4_poly_list, env5_poly_list]
    overall_init_list = [env1_init_list, env2_init_list, env3_init_list, env4_init_list, env5_init_list]
    overall_goal_list = [env1_goal_list, env2_goal_list, env3_goal_list, env4_goal_list, env5_goal_list]

    # Define environment
    rospy.init_node("train_ddpg_mov")
    env = GazeboEnvironment(laser_scan_half_num=laser_half_num, goal_near_th=goal_th, obs_near_th=obs_th,
                            obs_reward=obs_reward, goal_reward=goal_reward, goal_dis_amp=goal_dis_amp)

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
    net_dir = '../../training/save_model_weights/save_navigator/' \
              + 'HiDDPG_ctrl_actor_net_999' + '.pt'
    controller.actor_net.load_state_dict(torch.load(net_dir, map_location=lambda storage, loc: storage))
    controller.target_actor_net.load_state_dict(torch.load(net_dir, map_location=lambda storage, loc: storage))
    net_dir = '../../training/save_model_weights/save_navigator/' \
              + 'HiDDPG_ctrl_critic_net_999' + '.pt'
    controller.critic_net.load_state_dict(torch.load(net_dir, map_location=lambda storage, loc: storage))
    controller.target_critic_net.load_state_dict(torch.load(net_dir, map_location=lambda storage, loc: storage))


    # Define Tensorboard Writer
    tb_writer = SummaryWriter()

    # Define maximum steps per episode and reset maximum random action
    overall_steps = 0
    overall_episode = 999
    env_episode = 0
    env_ita = 4
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
    start_time = time.time()
    while True:
        episode_intrinsic_reward = 0
        ita_in_episode = 0
        done = False

        if env_ita < 4:
            robot_state = env.reset(env_episode, new_target_path=None)
            scaled_state = state_scale(robot_state)
        elif env_ita == 4:
            robot_state = env.reset(env_episode, target_path_list[env_episode])
            scaled_state = state_scale(robot_state)
        else:
            print("Wrong 'env_ita' value!")
        goal_dis, goal_dir = robot_2_goal_dis_dir(env.target_position, robot_state[0])
        scaled_dis = goal_dis if goal_dis != 0 else 0.3
        scaled_dis = 0.3 / scaled_dis
        scaled_dis = scaled_dis if scaled_dis <= 1 else 1
        scaled_dir = goal_dir / math.pi
        goal = np.array([scaled_dir, scaled_dis])
        joint_state_goal = np.concatenate((goal, scaled_state), axis=0)

        while not done and ita_in_episode < ita_per_episode:
            overall_steps += 1
            ita_in_episode += 1
            raw_action = controller.act(joint_state_goal)
            action = wheeled_network_2_robot_action(
                raw_action, linear_spd_max, linear_spd_min
            )
            next_robot_state, extrinsic_reward, done, \
                intrinsic_reward, intrinsic_done = env.step(action, env.target_position,
                                                            [-100, -100], [-100, -100],
                                                            [-100, -100], ita_in_episode)

            goal_dis, goal_dir = robot_2_goal_dis_dir(env.target_position, next_robot_state[0])
            scaled_dis = goal_dis if goal_dis != 0 else 0.3
            scaled_dis = 0.3 / scaled_dis
            scaled_dis = scaled_dis if scaled_dis <= 1 else 1
            scaled_dir = goal_dir / math.pi
            goal = np.array([scaled_dir, scaled_dis])

            scaled_next_state = state_scale(next_robot_state)
            next_joint_state_goal = np.concatenate((goal, scaled_next_state), axis=0)

            episode_intrinsic_reward += intrinsic_reward
            controller.remember(joint_state_goal, raw_action, intrinsic_reward, next_joint_state_goal, intrinsic_done)
            joint_state_goal = next_joint_state_goal

            # Train network with replay
            if len(controller.memory) > batch_size:
                actor_loss_value, critic_loss_value = controller.replay()
                tb_writer.add_scalar('Controller/actor_loss', actor_loss_value, overall_steps)
                tb_writer.add_scalar('Controller/critic_loss', critic_loss_value, overall_steps)
            tb_writer.add_scalar('Controller/action_epsilon', controller.epsilon, overall_steps)

            # Save Model
            if overall_steps % save_steps == 0:
                agent.save("../save_model_weights/save_navigator", controller.actor_net, overall_steps // save_steps, run_name + '_ctrl')

        print("Episode: {}/{}/{}, Controller Reward: {:.2f}, Steps: {}"
              .format(env_episode, overall_episode, episode_num, episode_intrinsic_reward, ita_in_episode + 1))
        tb_writer.add_scalar('Controller/reward', episode_intrinsic_reward, overall_steps)

        if ita_per_episode < iteration_num_max[env_ita]:
            ita_per_episode += iteration_num_step[env_ita]

        if overall_episode == 1499:
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

    train_ddpg_move(use_cuda=USE_CUDA)
