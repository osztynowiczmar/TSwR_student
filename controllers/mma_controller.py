import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.models = [
            ManiuplatorModel(Tp, m3 = 0.1, r3 = 0.05),
            ManiuplatorModel(Tp, m3 = 0.01, r3 = 0.01),
            ManiuplatorModel(Tp, m3 = 1.0, r3 = 0.3),
            ]
        
        self.i = 0

        self.prev_x = None
        self.prev_u = None
        self.Tp = Tp

        self.K_p = np.diag([100.0, 100.0])
        self.K_d = np.diag([10.0, 10.0])


    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)

        if self.prev_x is None or self.prev_u is None:
            self.i = 0
            return
        
        x = np.asarray(x).reshape(4,)
        x_prev = np.asarray(self.prev_x).reshape(4,)
        u_prev = np.asarray(self.prev_u).reshape(2,)

        q_prev = x_prev[0:2]
        q_dot_prev = x_prev[2:4]

        errors = []

        for model in self.models:
            M = model.M(x_prev)
            C = model.C(x_prev)

            q_ddot_pred = np.linalg.solve(M, u_prev - C @ q_dot_prev)
            q_dot_pred = q_dot_prev + self.Tp * q_ddot_pred
            
            q_pred = q_prev + self.Tp * q_dot_prev 
            x_pred = np.concatenate([q_pred, q_dot_pred])

            error = np.linalg.norm(x - x_pred)
            errors.append(error)
        self.i = int(np.argmin(errors))


    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        v = q_r_ddot + self.K_d @ (q_r_dot - q_dot) + self.K_p @ (q_r - q)
        #v = q_r_ddot

        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)

        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        u = np.asarray(u).reshape(2,)
        self.prev_x = x.copy()
        self.prev_u = u.copy()
        return u
