import numpy as np


class LinearDynamicalSystem:
    def __init__(self, F, H, Q, R, B=None):
        """
        Linear dynamical system.

        F - state-transition model
        H - observation model
        Q - covariance of process noise
        R - covariance of observation noise
        B - control-input model
            - assumed to be the identity matrix if not specified

        Assume state and control to be column vectors.
        Assume initial state to be zero vector.
        """
        dim_x = F.shape[0]
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.B = B
        self.x = np.zeros((dim_x, 1))

    def update(self, u=None):
        """
        Update system according to dynamics.

        u - control vector
            - assumed to be 0 if not specified
        """
        w = np.random.multivariate_normal(np.zeros(self.Q.shape[0]), self.Q).reshape(
            -1, 1
        )
        if not u:
            self.x = self.F @ self.x + w
        elif not self.B:
            self.x = self.F @ self.x + u + w
        else:
            self.x = self.F @ self.x + self.B @ u + w

    def observe(self):
        """
        Observe system according to dynamics.
        """
        v = np.random.multivariate_normal(np.zeros(self.R.shape[0]), self.R).reshape(
            -1, 1
        )
        return self.H @ self.x + v
