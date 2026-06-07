import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd


        self.p = p
        self.Tp = Tp

        A = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0]
        ])

        B = np.array([
            0.0,
            self.b,
            0.0
        ])

        L = np.array([
            3.0 * self.p,
            3.0 * self.p ** 2,
            self.p ** 3
        ])

        W = np.array([
            1.0,
            0.0,
            0.0
        ])


        self.eso = ESO(A, B, W, L, q0, Tp)

    def set_b(self, b):

        self.b = b

        B = np.array([
            0.0,
            self.b,
            0.0
        ])

        self.eso.set_B(B)

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q = float(np.asarray(x[0]).squeeze())

        z = self.eso.get_state()

        q_hat = z[0]
        q_dot_hat = z[1]
        f_hat = z[2]

        e = q_d - q_hat
        e_dot = q_d_dot - q_dot_hat

        v = q_d_ddot + self.kd * e_dot + self.kp * e
        u = (v - f_hat) / self.b

        self.eso.update(q, u)

        return float(u)
