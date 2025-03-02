import torch
from torch.utils import data
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os
import trajectory_generation as tg

'''
拍脑袋：gru-ddpg、lstm-seq2seq、kalman filter
小打磨：只预测1~2个点、更运动学的目标轨迹、合理的度量指标（成功率、实时性、鲁棒性、预测精度、画图看跟踪轨迹的好坏）、SAW 似乎也不适合验证轨迹预测？只要走到目标点附近，就有概率目标直接撞怀里
毕设：写论文、轨迹（更运动学的插值法）、方法（DDPG、MLP-Predictor、LSTM-Seq2Seq、Kalman Filter）、指标（成功率、实时性、鲁棒性、预测精度、画图看跟踪轨迹的好坏）
'''

# MLP Predictor
class MLP(nn.Module):
    def __init__(self, feature_dim, out_dim, hidden1=256, hidden2=256, hidden3=256):
        super(MLPPredictor, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, out_dim)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        x = self.relu(self.fc1(inputs))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = self.fc4(x)
        out = out.permute(0, 2, 1)
        return out

class MLPPredictor():
    def __init__(self, pred_tau, pred_length, traj_choice):
        self.pred_tau = pred_tau
        self.pred_length = pred_length
        self.model = MLP(self.pred_tau, self.pred_length)
        self.criterion = nn.SmoothL1Loss(reduction='mean')
        self.traj_choice = traj_choice

    def save(self, save_dir, pred_name):
        """
        Save Net weights
        :param save_dir: directory for saving weights
        :param pred_name: name of the predictor
        """
        try:
            os.mkdir(save_dir)
            print("Directory ", save_dir, " Created")
        except FileExistsError:
            print("Directory ", save_dir, " already exists")
        torch.save(self.model.state_dict(),
                   save_dir + '/' + pred_name + '_predictor' + '.pt')
        print("Model weights saved ...")

    def load_prediction_net(self, dir):
        """
        Load network for testing
        :param dir: directory of pt file
        """
        if self.traj_choice == 1:
            pred_name = "rose_predictor.pt"
        elif self.traj_choice == 2:
            pred_name = "spiral_predictor.pt"
        else:
            pred_name = "saw_predictor.pt"
        print(dir)
        if dir[-1] != '/':
            dir += '/'
        net_dir = dir + pred_name
        self.model.load_state_dict(torch.load(net_dir, map_location=lambda storage, loc: storage))
        return pred_name

    def window_generator(self, trajectory, tau, pred_length):
        '''
        Generate prediction window for data training & prediction
        :param trajectory(list):
        :param tau(int):
        :param pred_length(int):
        :return features, labels(numpy array, numpy array):
        '''
        window_size = tau + pred_length
        trajectory_n = len(trajectory)
        window = np.zeros((trajectory_n - window_size + 1, window_size, 2), dtype=np.float32)
        for i in range(window_size):
            window[:, i] = trajectory[i:trajectory_n - window_size + i + 1]
        features = window[:, :tau]
        labels = window[:, tau:]
        return features, labels

    def data_iter(self, N_trajectory, batch_size=84):
        init_pos = np.random.uniform(-9, 9, (N_trajectory, 2))
        traj = []
        if self.traj_choice == 1:
            for i in range(N_trajectory):
                traj.extend(tg.gen_rose_track(init_pos[i][0], init_pos[i][1], 1200))
        elif self.traj_choice == 2:
            for i in range(N_trajectory):
                traj.extend(tg.gen_spiral_track(init_pos[i][0], init_pos[i][1], 1200))
        else:
            for i in range(N_trajectory):
                traj.extend(tg.gen_rose_track(init_pos[i][0], init_pos[i][1], 1200))
        features, labels = self.window_generator(traj, self.pred_tau, self.pred_length)
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        dataset = data.TensorDataset(features, labels)
        dataiter = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataiter

    def train_model(self):
        N_traj = 3
        batch_size = 84
        train_iter = self.data_iter(N_traj, batch_size)
        criterion = self.criterion
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        epochs = 100
        for epoch in range(epochs):
            for i, data in enumerate(train_iter):
                inputs, targets = data
                optimizer.zero_grad()
                loss = criterion(self.model(inputs), targets)
                loss.backward()
                optimizer.step()

        # save model
        if self.traj_choice == 1:
            pred_name = "rose"
        elif self.traj_choice == 2:
            pred_name = "spiral"
        else:
            pred_name = "saw"
        self.save("../evaluation/saved_model", pred_name)

    def predict(self, traj_tau):
        traj_tau = torch.tensor(traj_tau, dtype=torch.float32).reshape(-1, self.pred_tau, 2)
        traj_pred = self.model(traj_tau)
        traj_pred = torch.squeeze(traj_pred, 0).detach().numpy().tolist()
        return traj_pred

if __name__ == '__main__':
    '''
    tunable parameters: 
    pred_tau, pred_length
    N_trajectory
    hidden[1,2,3]
    epoch
    batch_size
    '''
    '''
    △ Three predictors have been trained pretty well, DON'T CHANGE them anymore
    '''
    tau = 10
    pred_len = 15
    predictor_dir = '../evaluation/saved_model'
    predictor = TrajPredictor(tau, pred_len, 3)
    predictor.load_prediction_net(predictor_dir)

    # single-shot prediction
    init_pos = np.random.uniform(-9, 9, (2,))
    # traj = tg.gen_rose_track(init_pos[0], init_pos[1], 1200)
    # traj = tg.gen_spiral_track(init_pos[0], init_pos[1], 1200)
    traj = tg.gen_saw_track(init_pos[0], init_pos[1], 1005, 10)
    features, labels = predictor.window_generator(traj, tau, pred_len)
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    with torch.no_grad():
        oneshot_preds = predictor.model(features)
        test_loss = predictor.criterion(oneshot_preds, labels).item()
    print('Test Smooth L1 Loss: ', test_loss)
    # choose the last predicting step of every window as prediction result
    oneshot_preds = torch.cat((oneshot_preds[0, :, :], oneshot_preds[:, -1]), 0).detach().numpy()

    # test 'predict' function
    inputs = features.detach().numpy()
    predict_test = np.array(predictor.predict(inputs))
    # predict_test = np.concatenate((predict_test[0, :], predict_test[:, -1]), axis=0)
    # test for one group prediction
    period_inputs = inputs[500]
    period_labels = labels[500].detach().numpy()
    period_preds = predict_test[500]
    period_MSELoss = np.square(np.subtract(period_labels, period_preds)).mean()
    print('MSE Loss of selected period: ', period_MSELoss)

    # plot trajectory
    traj = np.array(traj)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(traj[:, 0], traj[:, 1], color='yellow', linewidth=1.2, s=0.5)    # true trajectory
    # single-shot prediciton
    # ax.scatter(oneshot_preds[0, 0], oneshot_preds[0, 1], color='limegreen', s=30)
    ax.scatter(oneshot_preds[:, 0], oneshot_preds[:, 1], color='lawngreen', linewidth=1.2, s=0.5)    # predicted trajectory
    # ax.text(oneshot_preds[0, 0]-1, oneshot_preds[0, 1], "start", size=12, ha="center", va="center")
    ax.plot(period_inputs[:, 0], period_inputs[:, 1], color='blue', linewidth=1.7)
    # plot target
    import matplotlib.patches as patches
    target = patches.RegularPolygon((period_labels[-1, 0], period_labels[-1, 1]), 4, 0.2*np.sqrt(2), color='gold')
    ax.add_patch(target)
    target = patches.RegularPolygon((period_inputs[-1, 0], period_inputs[-1, 1]), 4, 0.2 * np.sqrt(2), color='gold')
    ax.add_patch(target)

    ax.plot(period_labels[:, 0], period_labels[:, 1], color='orange', linewidth=1.7)
    ax.plot(period_preds[:, 0], period_preds[:, 1], color='red', linewidth=1.7)
    ax.scatter(period_preds[-1, 0], period_preds[-1, 1], color='orange', s=30)
    ax.set_xticks(np.linspace(-10, 10, 21))
    ax.set_yticks(np.linspace(-10, 10, 21))

    # # for 'predict' function test
    # fig2, ax2 = plt.subplots(figsize=(10,10))
    # ax2.plot(traj[:, 0], traj[:, 1], color='blue', linewidth=1.2)    # path
    # ax2.scatter(predict_test[:, 0], predict_test[:, 1], color='red', linewidth=1.2, s=0.5)
    # # ax2.scatter(traj[15:15+pred_len, 0], traj[15:15+pred_len, 1], color='k', linewidth=1.2, s=5)
    # ax2.set_xticks(np.linspace(-11, 11, 23))
    # ax2.set_yticks(np.linspace(-11, 11, 23))
    # ax2.grid()

    # plot obstacles

    poly_list, raw_poly_list = utility.gen_test_env_poly_list_env()
    for obs in raw_poly_list:
        x = [obs[num][0] for num in range(len(obs))]
        y = [obs[num][1] for num in range(len(obs))]
        x.append(obs[0][0])
        y.append(obs[0][1])
        ax.plot(x, y, 'k-')

    plt.show()