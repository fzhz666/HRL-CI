import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    """ Actor Network """

    def __init__(self, state_num, action_num, step_num, hidden1=256, hidden2=256, hidden3=256):
        """

        :param state_num: number of states
        :param action_num: number of actions
        :param step_num: number of steps
        :param hidden1: hidden layer 1 dimension
        :param hidden2: hidden layer 2 dimension
        :param hidden3: hidden layer 3 dimension
        """
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_num, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)

        # 分为两个部分的隐藏层
        self.fc3_action = nn.Linear(hidden2, hidden3)
        self.fc3_step = nn.Linear(hidden2, hidden3)

        # 输出 action_num 的全连接层
        self.fc_action = nn.Linear(hidden3, action_num)

        # 输出 step_num 的全连接层
        self.fc_step = nn.Linear(hidden3, step_num)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # 分别通过两个隐藏层
        x_action = self.relu(self.fc3_action(x))
        x_step = self.relu(self.fc3_step(x))

        # 分别通过两个全连接层输出 action_num 和 step_num
        out_action = self.sigmoid(self.fc_action(x_action))
        out_step = F.relu(self.fc_step(x_step))  # 使用 ReLU 作为示例

        return out_action, out_step


class CriticNet(nn.Module):
    """ Critic Network"""

    def __init__(self, state_num, action_num, step_num, hidden1=512, hidden2=512, hidden3=512):
        """

        :param state_num: number of states
        :param action_num: number of actions
        :param step_num: number of steps
        :param hidden1: hidden layer 1 dimension
        :param hidden2: hidden layer 2 dimension
        :param hidden3: hidden layer 3 dimension
        """
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_num, hidden1)

        # 两个不同的隐藏层
        self.fc2_action = nn.Linear(hidden1 + action_num, hidden2)
        self.fc2_step = nn.Linear(hidden1 + step_num, hidden2)

        self.fc3 = nn.Linear(hidden2 + hidden2, hidden3)

        # 输出 Critic 的值
        self.fc4 = nn.Linear(hidden3, 1)
        self.relu = nn.ReLU()

    def forward(self, xa):
        x, a, s = xa  # 输入包括状态 x，动作 a，和步数 s
        x = self.relu(self.fc1(x))

        # 分别连接 action 和 step 到隐藏层
        x_action = torch.cat([x, a], 1)
        x_step = torch.cat([x, s], 1)
        # print("before fc2 x_action dimensions:", x_action.size())  # x_action dimensions: torch.Size([256, 514])
        # print("before fc2 x_step dimensions:", x_step.size())  # x_step dimensions: torch.Size([256, 513])

        x_action = self.relu(self.fc2_action(x_action))
        x_step = self.relu(self.fc2_step(x_step))
        # print("x_action dimensions:", x_action.size())  # x_action dimensions: torch.Size([256, 512])
        # print("x_step dimensions:", x_step.size())   # x_step dimensions: torch.Size([256, 512])

        # 将两个不同类型的隐藏层结果合并
        x_combined = torch.cat([x_action, x_step], 1)
        # print("before fc3 x_combined dimensions:", x_combined.size())
        # before fc3 x_combined dimensions: torch.Size([256, 2048])

        x_combined = self.relu(self.fc3(x_combined))
        # print("x_combined dimensions:", x_combined.size())

        # 输出 Critic 的值
        out = self.fc4(x_combined)
        return out







