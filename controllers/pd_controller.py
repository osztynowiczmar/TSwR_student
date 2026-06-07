import numpy as np
from .controller import Controller


class PDDecentralizedController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.kp = kp
        self.kd = kd
        self.b = b

        self.p = p
        self.q0 = q0
        self.Tp = Tp

    def set_b(self, b):
        self.b = b

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q = float(np.asarray(x[0]).squeeze())
        q_dot = float(np.asarray(x[1]).squeeze())

        e = q_d - q
        e_dot = q_d_dot - q_dot

        v = q_d_ddot + self.kd * e_dot + self.kp * e

        u = v / self.b

        return float(u)