import numpy as np

#from models.free_model import FreeModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
#from models.ideal_model import IdealModel
from models.manipulator_model import ManiuplatorModel

class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.Tp = Tp
        self.model = ManiuplatorModel(Tp)

        self.Kp = self._as_gain_matrix(Kp)
        self.Kd = self._as_gain_matrix(Kd)

        p = np.asarray(p).reshape(-1)

        if p.size == 1:
            p = np.array([p.item(), p.item()])

        self.p = p

        L = np.array([
            [3.0 * p[0], 0.0],
            [0.0, 3.0 * p[1]],
            [3.0 * p[0]**2, 0.0],
            [0.0, 3.0 * p[1]**2],
            [p[0]**3, 0.0],
            [0.0, p[1]**3]
        ])

        W = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        ])

        A = np.zeros((6, 6))
        B = np.zeros((6, 2))
        self.eso = ESO(A, B, W, L, q0, Tp)

        q0 = np.asarray(q0).reshape(-1)

        if q0.size >= 4:
            self.update_params(q0[:2], q0[2:4])
        else:
            self.update_params(q0[:2], np.zeros(2))

    def _as_gain_matrix(self, gain):
        gain = np.asarray(gain)

        if gain.ndim == 0:
            return np.diag([float(gain), float(gain)])

        if gain.ndim == 1:
            return np.diag(gain)

        return gain

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        q = np.asarray(q).reshape(2,)
        q_dot = np.asarray(q_dot).reshape(2,)

        x = np.concatenate([q, q_dot])

        M_hat = self.model.M(x)
        C_hat = self.model.C(x)

        M_inv = np.linalg.inv(M_hat)

        A = np.zeros((6, 6))
        B = np.zeros((6, 2))

        A[0:2, 2:4] = np.eye(2)
        A[2:4, 2:4] = -M_inv @ C_hat
        A[2:4, 4:6] = np.eye(2)

        B[2:4, :] = M_inv

        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        x = np.asarray(x).reshape(4,)

        q = x[0:2]
        q_dot = x[2:4]

        q_d = np.asarray(q_d).reshape(2,)
        q_d_dot = np.asarray(q_d_dot).reshape(2,)
        q_d_ddot = np.asarray(q_d_ddot).reshape(2,)

        self.update_params(q, q_dot)

        z = self.eso.get_state()

        q_hat = z[0:2]
        q_dot_hat = z[2:4]
        f_hat = z[4:6]

        e = q_d - q_hat
        e_dot = q_d_dot - q_dot_hat

        v = q_d_ddot + self.Kd @ e_dot + self.Kp @ e

        M_hat = self.model.M(x)
        C_hat = self.model.C(x)

        u = M_hat @ (v - f_hat) + C_hat @ q_dot

        u = np.asarray(u).reshape(2,)

        self.eso.update(q, u)

        return u
