import numpy as np


def kalman_predict(x_hat, P, F, Q, B=None, u=None):
    """
    Calculate a priori estimation of state and estimate covariance.

    x_hat - state estimation
    P - estimate covariance matrix
    F - state-transition model
    Q - covariance of process noise
    B - control-input model
    u - control vector
    """
    if not u:
        x_hat_ = F @ x_hat
    elif not B:
        x_hat_ = F @ x_hat + u
    else:
        x_hat_ = F @ x_hat + B @ u

    P_ = F @ P @ F.T + Q
    return x_hat_, P_


def kalman_update(x_hat, P, F, H, Q, R, z):
    """
    Calculate a posteriori estimates of state and estimate covariance.

    x_hat - state estimation
    P - estimate covariance matrix
    F - state-transition model
    H - observation model
    Q - covariance of process noise
    R - covariance of observation noise
    z - observation from LDS
    """
    x_dim = x_hat.shape[0]

    # pre-fit residual
    y = z - H @ x_hat
    # pre-fit residual covariance
    S = H @ P @ H.T + R

    # Optimal Kalman gain
    K = P @ H.T @ np.linalg.inv(S)

    # a posteriori state estimate
    x_hat_ = x_hat + K @ y
    # a posteriori estimate covariance
    P_ = (np.identity(x_dim) - K @ H) @ P

    # post-fit residual
    y = z - H @ x_hat_
    return x_hat_, P_
