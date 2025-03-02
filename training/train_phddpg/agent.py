import numpy as np
import random
import os
import torch
import torch.nn as nn
from collections import deque
from ddpg_networks import ActorNet, CriticNet


class MetaController:
    '''
    Class for meta controller of agent

    Main function:
        1.
    '''
    def __init__(self,
                 state_num,
                 action_num,
                 traj_point_num,
                 goal_feature,
                 actor_net_dim=(256, 256, 256),
                 critic_net_dim=(512, 512, 512),
                 memory_size=1000,  # hDQN 是10^6
                 batch_size=128,
                 target_tau=0.01,
                 target_update_steps=5,
                 reward_gamma=1,
                 actor_lr=0.0001,  # hDQN 的上下层均是 0.00025
                 critic_lr=0.0001,
                 epsilon_start=0.9,
                 epsilon_end=0.01,
                 epsilon_decay=0.9997,
                 epsilon_decay_start=60000,
                 epsilon_decay_step=1,
                 use_cuda=True
                 ):
        '''

        :param state_num: number of state
        :param action_num: number of action
        :param traj_point_num: number of points along the predicted trajectory
        :param goal_feature: feature number of goal
        :param actor_net_dim: dimension of actor network
        :param critic_net_dim: dimension of critic network
        :param memory_size: size of memory
        :param batch_size: size of mini-batch
        :param target_tau: update rate for target network
        :param target_update_steps: update steps for target network
        :param reward_gamma: decay of future reward
        :param actor_lr: learning rate for actor network
        :param critic_lr: learning rate for critic network
        :param epsilon_start: max value for random action
        :param epsilon_end: min value for random action
        :param epsilon_decay: steps from max to min random action
        :param epsilon_decay_start: start step for epsilon start to decay
        :param epsilon_decay_step: steps between epsilon decay
        :param use_cuda: if or not use gpu
        '''
        self.state_num = state_num   # 20
        self.action_num = action_num    # 1
        self.traj_feature = traj_point_num * goal_feature   # 150*2
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_tau = target_tau
        self.target_update_steps = target_update_steps
        self.reward_gamma = reward_gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_start = epsilon_decay_start
        self.epsilon_decay_step = epsilon_decay_step
        self.use_cuda = use_cuda
        '''
        Random Action
        '''
        self.epsilon = epsilon_start
        '''
        Device
        '''
        if self.use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        '''
        Memory
        '''
        self.memory = deque(maxlen=self.memory_size)
        '''
        Networks and target networks
        '''
        self.actor_net = ActorNet(self.state_num + self.traj_feature, self.action_num,
                                  hidden1=actor_net_dim[0],
                                  hidden2=actor_net_dim[1],
                                  hidden3=actor_net_dim[2])
        self.critic_net = CriticNet(self.state_num + self.traj_feature, self.action_num,
                                    hidden1=critic_net_dim[0],
                                    hidden2=critic_net_dim[1],
                                    hidden3=critic_net_dim[2])
        self.target_actor_net = ActorNet(self.state_num + self.traj_feature, self.action_num,
                                         hidden1=actor_net_dim[0],
                                         hidden2=actor_net_dim[1],
                                         hidden3=actor_net_dim[2])
        self.target_critic_net = CriticNet(self.state_num + self.traj_feature, self.action_num,
                                           hidden1=critic_net_dim[0],
                                           hidden2=critic_net_dim[1],
                                           hidden3=critic_net_dim[2])

        # # 获取 self.fc3 层的权重
        # fc3_weights = self.critic_net.fc3.weight
        # # 打印权重维度
        # print("fc3 weights dimensions:", fc3_weights.size())  # fc3 weights dimensions: torch.Size([512, 512])
        #
        # fc4_weights = self.critic_net.fc4.weight
        # # 打印权重维度
        # print("fc4 weights dimensions:", fc4_weights.size())  # fc4 weights dimensions: torch.Size([1, 512])

        _hard_update(self.target_actor_net, self.actor_net)
        _hard_update(self.target_critic_net, self.critic_net)
        self.actor_net.to(self.device)
        self.critic_net.to(self.device)
        self.target_actor_net.to(self.device)
        self.target_critic_net.to(self.device)
        '''
        Criterion and optimizers
        '''
        self.criterion = nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.critic_lr)
        '''
        Step counter
        '''
        self.step_ita = 0

    def select_goal(self, joint_state_traj, explore=True, train=True):
        """
        Generate goal selection for controller based on state and trajectory
        :param joint_state_traj: vector jointing current state and current goal trajectory
        :param explore: if or not do random explore
        :param train: if or not in training
        :return: goal selection among trajectory for controller
        """
        with torch.no_grad():
            joint_state_traj = np.array(joint_state_traj)
            joint_state_traj = torch.Tensor(joint_state_traj.reshape((1, -1))).to(self.device)
            selection = self.actor_net(joint_state_traj).to('cpu')
            selection = selection.numpy().squeeze()
        if train:
            if self.step_ita > self.epsilon_decay_start and self.epsilon > self.epsilon_end:
                if self.step_ita % self.epsilon_decay_step == 0:
                    self.epsilon = self.epsilon * self.epsilon_decay
            noise = np.random.normal(0, 0.5) * self.epsilon
            selection = noise + selection
            selection = np.clip(selection, 0., 1.)
        elif explore:
            noise = np.random.normal(0, 0.5) * self.epsilon_end
            selection = noise + selection
            selection = np.clip(selection, 0., 1.)
        return np.float32(selection)

    def select_goal_step(self, joint_state_traj, explore=True, train=True):
        """
        Generate goal selection for controller based on state and trajectory
        :param joint_state_traj: vector jointing current state and current goal trajectory
        :param explore: if or not do random explore
        :param train: if or not in training
        :return: goal selection among trajectory for controller
        """
        with torch.no_grad():
            joint_state_traj = np.array(joint_state_traj)
            joint_state_traj = torch.Tensor(joint_state_traj.reshape((1, -1))).to(self.device)
            goal_step = self.actor_net(joint_state_traj).to('cpu')
            goal_step = goal_step.numpy().squeeze()
        if train:
            if self.step_ita > self.epsilon_decay_start and self.epsilon > self.epsilon_end:
                if self.step_ita % self.epsilon_decay_step == 0:
                    self.epsilon = self.epsilon * self.epsilon_decay
            noise = np.random.normal(0, 0.5) * self.epsilon
            goal_step = noise + goal_step
            goal_step = np.clip(goal_step, [0., 0.], [1., 1.])
        elif explore:
            noise = np.random.normal(0, 0.5) * self.epsilon_end
            goal_step = noise + goal_step
            goal_step = np.clip(goal_step, [0., 0.], [1., 1.])
        return np.float32(goal_step[0]), np.float32(goal_step[1])


    def remember(self, state, action, reward, next_state, done):
        """
        Add New Memory Entry into memory deque
        :param state: current state
        :param action: current action
        :param reward: reward after action
        :param next_state: next action
        :param done: if is done
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Experience Replay Training
        :return: actor_loss_item, critic_loss_item
        """
        state_batch, action_batch, reward_batch, nx_state_batch, done_batch = _random_minibatch(self.memory, self.batch_size, self.device)
        '''
        Compuate Target Q Value
        '''
        with torch.no_grad():
            nx_action_batch = self.target_actor_net(nx_state_batch)
            next_q = self.target_critic_net([nx_state_batch, nx_action_batch])
            target_q = reward_batch + self.reward_gamma * next_q * (1. - done_batch)
        '''
        Update Critic Network
        '''
        self.critic_optimizer.zero_grad()
        current_q = self.critic_net([state_batch, action_batch])
        critic_loss = self.criterion(current_q, target_q)
        critic_loss_item = critic_loss.item()
        critic_loss.backward()
        self.critic_optimizer.step()
        '''
        Update Actor Network
        '''
        self.actor_optimizer.zero_grad()
        current_action = self.actor_net(state_batch)
        actor_loss = -self.critic_net([state_batch, current_action])
        actor_loss = actor_loss.mean()
        actor_loss_item = actor_loss.item()
        actor_loss.backward()
        self.actor_optimizer.step()
        '''
        Update Target Networks
        '''
        self.step_ita += 1
        if self.step_ita % self.target_update_steps == 0:
            _soft_update(self.target_actor_net, self.actor_net, self.target_tau)
            _soft_update(self.target_critic_net, self.critic_net, self.target_tau)
        return actor_loss_item, critic_loss_item

    def reset_epsilon(self, new_epsilon, new_decay):
        """
        Set Epsilon to a new value
        :param new_epsilon: new epsilon value
        :param new_decay: new epsilon decay
        """
        self.epsilon = new_epsilon
        self.epsilon_decay = new_decay


class Controller:
    '''
    Class for controller of agent

    '''
    def __init__(self,
                 state_num,
                 goal_feature,
                 action_num,
                 actor_net_dim=(256, 256, 256),
                 critic_net_dim=(512, 512, 512),
                 memory_size=1000,  # hDQN 是10^6
                 batch_size=128,
                 target_tau=0.01,
                 target_update_steps=5,
                 reward_gamma=0.99,
                 actor_lr=0.0001,  # hDQN 的上下层均是 0.00025
                 critic_lr=0.0001,
                 epsilon_start=0.9,
                 epsilon_end=0.01,
                 epsilon_decay=0.999,
                 epsilon_decay_start=60000,
                 epsilon_decay_step=1,
                 use_cuda=True
                 ):
        '''

        :param state_num: number of state
        :param goal_feature: feature number of goal
        :param action_num: number of action
        :param actor_net_dim: dimension of actor network
        :param critic_net_dim: dimension of critic network
        :param memory_size: size of memory
        :param batch_size: size of mini-batch
        :param target_tau: update rate for target network
        :param target_update_steps: update steps for target network
        :param reward_gamma: decay of future reward
        :param actor_lr: learning rate for actor network
        :param critic_lr: learning rate for critic network
        :param epsilon_start: max value for random action
        :param epsilon_end: min value for random action
        :param epsilon_decay: steps from max to min random action
        :param epsilon_decay_start: start step for epsilon start to decay
        :param epsilon_decay_step: steps between epsilon decay
        :param use_cuda: if or not use gpu
        '''
        self.state_num = state_num
        self.goal_feature = goal_feature
        self.action_num = action_num
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_tau = target_tau
        self.target_update_steps = target_update_steps
        self.reward_gamma = reward_gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_start = epsilon_decay_start
        self.epsilon_decay_step = epsilon_decay_step
        self.use_cuda = use_cuda
        '''
        Random Action
        '''
        self.epsilon = epsilon_start
        '''
        Device, reuse
        '''
        if self.use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        '''
        Memory
        '''
        self.memory = deque(maxlen=self.memory_size)
        '''
        Networks and target networks
        '''
        self.actor_net = ActorNet(self.goal_feature + self.state_num, self.action_num,
                                  hidden1=actor_net_dim[0],
                                  hidden2=actor_net_dim[1],
                                  hidden3=actor_net_dim[2])
        self.critic_net = CriticNet(self.goal_feature + self.state_num, self.action_num,
                                    hidden1=critic_net_dim[0],
                                    hidden2=critic_net_dim[1],
                                    hidden3=critic_net_dim[2])
        self.target_actor_net = ActorNet(self.goal_feature + self.state_num, self.action_num,
                                         hidden1=actor_net_dim[0],
                                         hidden2=actor_net_dim[1],
                                         hidden3=actor_net_dim[2])
        self.target_critic_net = CriticNet(self.goal_feature + self.state_num, self.action_num,
                                           hidden1=critic_net_dim[0],
                                           hidden2=critic_net_dim[1],
                                           hidden3=critic_net_dim[2])
        _hard_update(self.target_actor_net, self.actor_net)
        _hard_update(self.target_critic_net, self.critic_net)
        self.actor_net.to(self.device)
        self.critic_net.to(self.device)
        self.target_actor_net.to(self.device)
        self.target_critic_net.to(self.device)
        '''
        Criterion and optimizers
        '''
        self.criterion = nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.critic_lr)
        '''
        Step counter, about epsilon, check do I need to reuse or specify respectively
        '''
        self.step_ita = 0

    def act(self, joint_state_goal, explore=True, train=True):
        """
        Generate action based on state and goal
        :param joint_state_goal: vector jointing current state and current goal
        :param explore: if or not do random explore
        :param train: if or not in training
        :return: goal selection among trajectory for controller
        """
        with torch.no_grad():
            joint_state_goal = np.array(joint_state_goal)
            joint_state_goal = torch.Tensor(joint_state_goal.reshape((1, -1))).to(self.device)
            action = self.actor_net(joint_state_goal).to('cpu')
            action = action.numpy().squeeze()
        if train:
            if self.step_ita > self.epsilon_decay_start and self.epsilon > self.epsilon_end:
                if self.step_ita % self.epsilon_decay_step == 0:
                    self.epsilon = self.epsilon * self.epsilon_decay
            noise = np.random.randn(self.action_num) * self.epsilon
            action = noise + (1 - self.epsilon) * action
            # noise = np.random.normal(0, 0.5, self.action_num) * self.epsilon
            # action = noise + action
            action = np.clip(action, [0., 0.], [1., 1.])
        elif explore:
            noise = np.random.randn(self.action_num) * self.epsilon_end
            action = noise + (1 - self.epsilon_end) * action
            # noise = np.random.normal(0, 0.5, self.action_num) * self.epsilon_end
            # action = noise + action
            action = np.clip(action, [0., 0.], [1., 1.])
        return action.tolist()

    def remember(self, state, action, reward, next_state, done):
        """
        Add New Memory Entry into memory deque
        :param state: current state
        :param action: current action
        :param reward: reward after action
        :param next_state: next action
        :param done: if is done
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Experience Replay Training
        :return: actor_loss_item, critic_loss_item
        """
        state_batch, action_batch, reward_batch, nx_state_batch, done_batch = _random_minibatch(self.memory, self.batch_size, self.device)
        '''
        Compuate Target Q Value
        '''
        with torch.no_grad():
            nx_action_batch = self.target_actor_net(nx_state_batch)
            next_q = self.target_critic_net([nx_state_batch, nx_action_batch])
            target_q = reward_batch + self.reward_gamma * next_q * (1. - done_batch)
        '''
        Update Critic Network
        '''
        self.critic_optimizer.zero_grad()
        current_q = self.critic_net([state_batch, action_batch])
        critic_loss = self.criterion(current_q, target_q)
        critic_loss_item = critic_loss.item()
        critic_loss.backward()
        self.critic_optimizer.step()
        '''
        Update Actor Network
        '''
        self.actor_optimizer.zero_grad()
        current_action = self.actor_net(state_batch)
        actor_loss = -self.critic_net([state_batch, current_action])
        actor_loss = actor_loss.mean()
        actor_loss_item = actor_loss.item()
        # if self.step_ita % 10 == 0:
        #     current_action.register_hook(lambda grad: print("\ngrad of action:\n\t", grad.mean(), "\t", grad.max(), "\t", grad.min()))
        actor_loss.backward()
        # if self.step_ita % 10 == 0:
            # print("\ngrad of actor net:")
            # for name, weight in self.actor_net.named_parameters():
                # if weight.requires_grad:
                    # print(name, ":\t", weight.grad.mean(), "\t", weight.grad.max(), "\t", weight.grad.min())
        self.actor_optimizer.step()
        '''
        Update Target Networks
        '''
        self.step_ita += 1
        if self.step_ita % self.target_update_steps == 0:
            _soft_update(self.target_actor_net, self.actor_net, self.target_tau)
            _soft_update(self.target_critic_net, self.critic_net, self.target_tau)
        return actor_loss_item, critic_loss_item

    def reset_epsilon(self, new_epsilon, new_decay):
        """
        Set Epsilon to a new value
        :param new_epsilon: new epsilon value
        :param new_decay: new epsilon decay
        """
        self.epsilon = new_epsilon
        self.epsilon_decay = new_decay


def save(save_dir, actor_net, episode, run_name):
    """
    Save Actor Net weights
    :param save_dir: directory for saving weights
    :param episode: number of episode
    :param run_name: name of the run
    """
    try:
        os.makedirs(save_dir)
        print("Directory ", save_dir, " Created")
    except FileExistsError:
        print("Directory", save_dir, " already exists")
    torch.save(actor_net.state_dict(),
               save_dir + '/' + run_name + '_actor_net_s' + str(episode) + '.pt')
    print("Episode " + str(episode) + " weights saved ...")


def _random_minibatch(memory, batch_size, device):
    """
    Random select mini-batch from memory
    :return: state batch, action batch, reward batch, next state batch, done batch
    """
    state_batch, action_batch, reward_batch, nx_state_batch, done_batch = zip(*random.sample(memory, batch_size))
    state_batch = torch.Tensor(state_batch).reshape(batch_size, -1).to(device)
    action_batch = torch.Tensor(action_batch).reshape(batch_size, -1).to(device)
    reward_batch = torch.Tensor(reward_batch)
    reward_batch = reward_batch.view((batch_size, 1)).to(device)
    nx_state_batch = torch.Tensor(nx_state_batch).reshape(batch_size, -1).to(device)
    done_batch = torch.Tensor(done_batch)
    done_batch = done_batch.view((batch_size, 1)).to(device)
    return state_batch, action_batch, reward_batch, nx_state_batch, done_batch


def _hard_update(target, source):
    """
    Hard update weights from source network to target network
    :param target: target network
    :param source: source network
    """
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def _soft_update(target, source, target_tau):
    """
    Soft update weights from source network to target network
    :param target: target network
    :param source: source network
    """
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - target_tau) + param.data * target_tau
            )
