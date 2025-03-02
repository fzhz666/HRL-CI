import torch
from torch.utils import data
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from trajectory_generation import dimer

def interpolation(x, y, T=1000):
    interval = 1
    N = len(x)
    t = np.linspace(0, (N - 1) * interval, T)
    x_contin, y_contin = 0, 0
    for n in range(N):
        x_contin += x[n] * np.sinc(t / interval - n)
        y_contin += y[n] * np.sinc(t / interval - n)
    return x_contin, y_contin

def window_generator(trajectory, tau, pred_length):
    '''
    Generate prediction window for data training & prediction
    :param trajectory():
    :param tau(int):
    :param pred_length(int):
    :return features, labels():
    '''
    window_size = tau + pred_length
    window = torch.zeros(T - window_size + 1, 2, window_size)
    for i in range(window_size):
        window[:, :, i] = trajectory[i:T - window_size + i + 1]
    features = window[:, :, :tau]
    labels = window[:, :, tau:]
    return features, labels

# MLP Predictor
class MLPPredictor(nn.Module):
    '''
    用 MLP 进行时序预测时，两种不同的数据组织方式，仅仅是一个微小差别，都能带来差异巨大的结果
    结果就是（N, feature_in, L）比 （N, L, feature_in）+ Flatten 的效果好太多多多多多多了，根本没得比
    若用 LSTM/GRU 来预测（N, L, feature_in）形式的数据，泛化效果也会很差（ LSTM 效果差的原因终于找到了/(ㄒoㄒ)/~~ ）
    '''
    def __init__(self, feature_dim, out_dim, hidden1=16, hidden2=16, hidden3=16):
        super(MLPPredictor, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, out_dim)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.fc1(inputs))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = self.fc4(x)
        return out

# trajectory generation
T = 1000
init_pos = [0, 0]
traj_x, traj_y = [], []
# # flower
# for i in range(T):
#     k = i * np.pi / 500
#     r = 8 * np.sin(4 * k)
#     traj_x.append(0.5 * init_pos[0] + r * np.cos(k))
#     traj_y.append(0.5 * init_pos[1] + r * np.sin(k))

# # spiral
# for i in range(T):
#     k = i * math.pi / 90
#     r = 0.23 * k
#     traj_x.append(0.5 * init_pos[0] + r * math.cos(k))
#     traj_y.append(0.5 * init_pos[1] + r * math.sin(k))

# # sine-circle
# for i in range(1440):
#     k = (i / 720) * math.pi
#     r = 0.65 * math.sqrt(init_pos[0] ** 2 + init_pos[1] ** 2) + 0.7 * math.sin(16 * k)
#     x.append(r * math.sin(k + math.atan2(init_pos[0], init_pos[1])))
#     y.append(r * math.cos(k + math.atan2(init_pos[0], init_pos[1])))

# saw
x, y = dimer(69, init_pos[0], init_pos[1])
traj_x, traj_y = interpolation(x, y, T)

traj = torch.tensor(np.stack((traj_x, traj_y), axis=1), dtype=torch.float32)

# build dataset and training iterator
tau = 10
pred_len = 1
features, labels = window_generator(traj, tau, pred_len)
batch_size, n_train = 84, 600
trainset = data.TensorDataset(features[:n_train], labels[:n_train])
train_iter = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# define MLP net
net = MLPPredictor(tau, pred_len)

criterion = nn.SmoothL1Loss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# train MLP net
epochs = 7
running_loss = 0.0
for epoch in range(epochs):
    for i, data in enumerate(train_iter):
        inputs, targets = data
        optimizer.zero_grad()
        loss = criterion(net(inputs), targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f'[{epoch + 1}, {i + 1}] loss: {running_loss:.3f}')
        running_loss = 0.0

print('MLP Training finished')

# single-shot prediction
# onestep_preds = net(features).detach().numpy()
with torch.no_grad():
    oneshot_preds = net(features)
    total_loss = criterion(oneshot_preds, labels).item()
print('Test Smooth L1 Loss: ', total_loss)
# choose the last predicting step of every window as prediction result
oneshot_preds = torch.cat((oneshot_preds[0, :, :].permute(1, 0), oneshot_preds[:, :, -1]), 0).detach().numpy()

# muti-step rolling prediction
# pred_steps = 5
# multistep_preds = traj[0:T-pred_steps].reshape(-1, tau, 2)
# multistep_preds = torch.cat((multistep_preds, torch.zeros_like(multistep_preds)), 1)
# for step in range(pred_steps):
#     multistep_preds[:, tau+step] = net(multistep_preds[:, step:tau+step])
# multistep_preds = multistep_preds[:, tau:].reshape(-1, 2).detach().numpy()

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(traj_x, traj_y, color='#4169E1', linestyle='-.', linewidth=1.2, label='True trajectory')    # path
# single-shot prediciton
ax.plot(oneshot_preds[:, 0], oneshot_preds[:, 1], color='red', linewidth=1.2, label='Predicted trajectory')
ax.scatter(oneshot_preds[0, 0], oneshot_preds[0, 1], color='limegreen', s=50)
ax.text(oneshot_preds[0, 0]-1, oneshot_preds[0, 1], "start", size=13, ha="center", va="center")
ax.scatter(oneshot_preds[600, 0], oneshot_preds[600, 1], color='orange', s=50)
ax.text(oneshot_preds[600, 0]-1.5, oneshot_preds[600, 1]-0.5, "test start", size=13, ha="center", va="center")
ax.scatter(oneshot_preds[-1, 0], oneshot_preds[-1, 1], color='r', s=50)
ax.text(oneshot_preds[-1, 0]-1.5, oneshot_preds[-1, 1]-0.5, "end", size=13, ha="center", va="center")
# multi-step prediction plot
# ax.scatter(multistep_preds[0, 0], multistep_preds[0, 1], color='limegreen', s=30)
# ax.plot(multistep_preds[:, 0], multistep_preds[:, 1], color='red', linewidth=1.5)
# ax.text(multistep_preds[0, 0]-1, multistep_preds[0, 1], "start", size=12, ha="center", va="center")
ax.set_xlim((-11, 11))
ax.set_ylim((-11, 11))
ax.set_aspect('equal', 'box')
ax.legend()
plt.show()