import numpy as np
from numpy.linalg import inv

class KalmanFilter:
    def __init__(self):
        '''
        :param n: state dimension (6 in our case)
        :param m: measurement dimension (2 in our case)
        :param P0: initial process covariance matrix
        :param Q: process error covariance matrix
        '''
        dt = 0.1
        # transition matrix  x  x' y  y' x'' y''
        self.F = np.array([[1, 1 * dt, 0, 0, 0.5 * dt * dt, 0],  # x
                           [0, 1, 0, 0, 1 * dt, 0],  # x'
                           [0, 0, 1, 1 * dt, 0, 0.5 * dt * dt],  # y
                           [0, 0, 0, 1, 0, 1 * dt],  # y'
                           [0, 0, 0, 0, 1, 0],  # x''
                           [0, 0, 0, 0, 0, 1]])  # y''
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0]])
        self.m = self.H.shape[0]
        self.n = self.H.shape[1]
        self.K = np.zeros((self.n, self.m))
        self.P = np.diag(np.full(self.n, 15))
        # a too small R will make KF too sensitive to every direction changing, which is the effect of KF as a predictor
        # a too big R (or too small Q) will make KF getting delayed of direction changing prediction
        self.R = np.diag(np.full(self.m, 0.1))  # 0.02 0.0001
        # self.Q = np.diag(np.full(self.n, 0.0001))  # 0.03 0.1
        sigma_a = 0.01
        self.Q = sigma_a * np.array([[0.25 * dt**4, 0.5 * dt**3, 0, 0, 0.5 * dt**2, 0],
                                     [0.5 * dt**3, dt**2, 0, 0, dt, 0],
                                     [0, 0, 0.25 * dt**4, 0.5 * dt**3, 0, 0.5 * dt**2],
                                     [0, 0, 0.5 * dt**3, dt**2, 0, dt],
                                     [0.5 * dt**2, dt, 0, 0, 1, 0],
                                     [0, 0, 0.5 * dt**2, dt, 0, 1]])

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

class KFPredictor:
    def __init__(self, pred_tau, pred_length, traj_max_length=1500):
        self.pred_tau = pred_tau
        self.pred_length = pred_length
        self.kf = KalmanFilter()
        self.state = np.zeros((traj_max_length, self.kf.F.shape[0]))

    def predict(self, traj_observed, traj_ita):
        '''
        self.state[traj_ita] refers to the same trajectory point as traj_observed[-1]
        '''
        predict_start = traj_ita + 1
        traj_observed = np.array(traj_observed)

        if traj_ita == 0:
            x0, y0 = traj_observed[0]
            self.state[0] = np.array([x0, 0, y0, 0, 0, 0])
        elif traj_ita == 1:
            x0, y0 = traj_observed[0]
            vx0 = (traj_observed[1, 0] - x0) * 1.0
            vy0 = (traj_observed[1, 1] - y0) * 1.0
            self.state[0] = np.array([x0, vx0, y0, vy0, 0, 0])

        observe_num = min(predict_start, self.pred_tau)
        state_ita = max(0, predict_start - self.pred_tau)
        for i in range(observe_num + self.pred_length):
            state_prior = self.kf.predict(self.state[state_ita])
            if i < observe_num - 1:
                measurement = traj_observed[i + 1]
                self.state[state_ita + 1] = self.kf.correction(state_prior, measurement)
            else:
                self.state[state_ita + 1] = state_prior
            state_ita += 1

        # check the filtered observed trajectory
        # traj_filtered = self.state[predict_start - observe_num: predict_start, [0, 2]]
        traj_predict = self.state[predict_start: predict_start + self.pred_length, [0, 2]]

        return traj_predict