import numpy as np
from numpy.linalg import inv

class KalmanFilter:
    def __init__(self):
        '''
        :param n: state dimension (4 in our case)
        :param m: measurement dimension (2 in our case)
        :param P0: initial process covariance matrix
        :param Q: process error covariance matrix
        '''
        self.init_matrix('acceleration')
        self.m = self.H.shape[0]
        self.n = self.H.shape[1]
        self.K = np.zeros((self.n, self.m))
        self.P = np.diag(np.full(self.n, 15))
        self.R = np.diag(np.full(self.m, 0.02))
        self.Q = np.diag(np.full(self.n, 0.03))

    def init_matrix(self, model_choice, dt=0.1):
        if model_choice == 'acceleration':  # use acceleration
            # transition matrix  x  x' y  y' x'' y''
            self.F = np.array([[1, 1 * dt, 0, 0, 0.5 * dt * dt, 0],  # x
                          [0, 1, 0, 0, 1 * dt, 0],  # x'
                          [0, 0, 1, 1 * dt, 0, 0.5 * dt * dt],  # y
                          [0, 0, 0, 1, 0, 1 * dt],  # y'
                          [0, 0, 0, 0, 1, 0],  # x''
                          [0, 0, 0, 0, 0, 1]])  # y''

            self.H = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0]])
        else:
            self.F = np.array([[1, 1 * dt, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 1 * dt],
                          [0, 0, 0, 1]])

            self.H = np.array([[1, 0, 0, 0],  # m x n     m = 2, n = 4 or 6
                          [0, 0, 1, 0]])

    def predict(self, x_previous):
        '''
        prediction step:
        x_prior = F x_previous
        P_prior = F P_previous F^T + Q

        :param n: state dimension (4 in our case)
        :param m: measurement dimension (2 in our case)
        :param P0: initial process covariance matrix
        :param Q: process error covariance matrix
        :return: new a posteriori prediction
        '''
        x_prior = np.dot(self.F, x_previous)    # update previous state estimate with state transition matrix
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q    # this is P minus
        return x_prior

    def correction(self, x_prior, z):
        '''
        correction step, must be used immediately after prediction step:
        D = H P_prior H^T + R
        K = P_prior H^T D^-1
        e = z - H x_prior
        x = x_prior + K e
        P = (I - K H) P_prior

        :param n: state dimension (4 in our case)
        :param m: measurement dimension (2 in our case)
        :param P0: initial process covariance matrix
        :param Q: process error covariance matrix
        :return: new a posteriori prediction
        '''
        P_prior = self.P

        inv_D = inv(np.dot(np.dot(self.H, P_prior), self.H.T) + self.R)
        self.K = np.dot(np.dot(P_prior, self.H.T), inv_D)

        e = z - np.dot(self.H, x_prior)
        x = x_prior + np.dot(self.K, e)

        self.P = np.dot((np.eye(self.n) - np.dot(self.K, self.H)), P_prior)

        return x

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import trajectory_generation as tg
    import matplotlib.patches as patches
    import pickle

    # plot obstacles
    from evaluation.eval_simulation import utils
    fig, ax = plt.subplots(figsize=(10, 10))

    poly_list, raw_poly_list = utils.gen_test_env_poly_list_env()
    for obs in raw_poly_list:
        x = [obs[num][0] for num in range(len(obs))]
        y = [obs[num][1] for num in range(len(obs))]
        x.append(obs[0][0])
        y.append(obs[0][1])
        ax.plot(x, y, 'k-')
    ax.set_xticks(np.linspace(-10, 10, 21))
    ax.set_yticks(np.linspace(-10, 10, 21))

    # good performance on short-term prediction, bad on long-term prediction
    tau = 30
    pred_len = 150

    # generate trajectory for test
    init_pos = np.random.uniform(-9, 9, (2,))
    # traj = tg.gen_rose_track(init_pos[0], init_pos[1], 1005)
    # traj = tg.gen_spiral_track(init_pos[0], init_pos[1], 1005)
    # traj = tg.gen_saw_track(init_pos[0], init_pos[1], 1005, 9)
    # traj = tg.gen_square_track(0, 0)
    # traj = tg.gen_random_track(init_pos[0], init_pos[1], poly_list)
    traj = tg.gen_random_rrt_track(init_pos[0], init_pos[1], 1010, ((-10, 10), (-10, 10)), poly_list)

    traj = np.array(traj)
    traj_tau = traj[: tau]
    traj_label = traj[tau: tau + pred_len]
    pred_traj = np.zeros_like(traj)

    # state initialization
    kf = KalmanFilter()
    state = np.zeros((traj.shape[0], kf.F.shape[0]))
    x0, y0 = traj[0]
    vx0 = (traj[1, 0] - x0) * 10
    vy0 = (traj[1, 1] - y0) * 10
    state[0] = np.array([x0, vx0, y0, vy0, 0, 0])
    pred_traj[0] = traj[0]

    # # predict a single period
    # i = 0
    # while i < tau + pred_len:
    #     state_prior = kf.predict(state[i])
    #     if i < tau - 1:
    #         measurement = traj[i + 1]
    #         state[i + 1] = kf.correction(state_prior, measurement)
    #     else:
    #         state[i + 1] = state_prior
    #     i = i + 1
    #
    # pred_traj = state[: tau+pred_len, [0, 2]]

    # compute accuracy through MSE Loss
    # MSELoss = np.square(np.subtract(traj[: tau+pred_len], pred_traj)).mean()
    # print('MSE Loss: ', MSELoss)

    # # plot trajectory
    # ax.plot(traj_tau[:, 0], traj_tau[:, 1], color='blue', linewidth=1, marker='o', markersize=1.6)
    # ax.plot(traj_label[:, 0], traj_label[:, 1], color='lawngreen', linewidth=1, marker='o', markersize=1.2)
    # ax.plot(pred_traj[:, 0], pred_traj[:, 1], color='red', linewidth=0.5, marker='o', markersize=0.8)
    # ax.set_xticks(np.linspace(-10, 10, 21))
    # ax.set_yticks(np.linspace(-10, 10, 21))

    # rolling prediction
    # 正式写时记得加注释
    images = []
    i = 0
    pred_start = tau
    while pred_start < traj.__len__():
        state_prior = kf.predict(state[i])
        if i < pred_start - 1:
            measurement = traj[i + 1]
            state[i + 1] = kf.correction(state_prior, measurement)
        else:
            state[i + 1] = state_prior

        if i == pred_start + pred_len - 1:
            pred_traj = state[pred_start: pred_start + pred_len, [0, 2]]
            # compute MSE Loss
            MSELoss = np.square(np.subtract(traj[pred_start: pred_start+pred_len], pred_traj)).mean()
            print('MSE Loss: ', MSELoss)
            # plot
            image1, = ax.plot(traj[: pred_start, 0], traj[: pred_start, 1], color='skyblue', linewidth=1, marker='o', markersize=1.6)
            image2, = ax.plot(traj[pred_start-tau: pred_start, 0], traj[pred_start-tau: pred_start, 1], color='royalblue', linewidth=1, marker='o', markersize=1.6)
            image3, = ax.plot(traj[pred_start: pred_start+pred_len, 0], traj[pred_start: pred_start+pred_len, 1], color='limegreen', linewidth=1, marker='o', markersize=1.6)
            image4, = ax.plot(pred_traj[:, 0], pred_traj[:, 1], color='red', linewidth=0.5, marker='o', markersize=0.8)
            # plot target
            target = patches.RegularPolygon((traj[pred_start, 0], traj[pred_start, 1]), 4, 0.2 * np.sqrt(2),
                                            color='tomato')
            target_img = ax.add_patch(target)
            title = ax.text(0.5, 1.05, "MSE = {:.4f}".format(MSELoss),
                            size=plt.rcParams["axes.titlesize"],
                            ha="center", transform=ax.transAxes, )
            images.append([image1, image2, image3, image4, target_img, title])

            pred_start = pred_start + 1
            i = pred_start - tau - 1

        i = i + 1

    ani = animation.ArtistAnimation(fig, images, interval=50, blit=False)
    plt.show()