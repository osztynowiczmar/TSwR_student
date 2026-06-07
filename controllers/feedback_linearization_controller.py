import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)

        self.K_p = np.diag([150.0, 150.0])
        self.K_d = np.diag([25.0, 25.0])

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """

        q_dot = np.asarray(x[2:4]).reshape(2,)
        v = np.asarray(q_r_ddot).reshape(2,)
        u = self.model.M(x) @ v + self.model.C(x) @ q_dot

        #Zadanie8
        q = np.asarray(x[0:2]).reshape(2,)
        q_dot = np.asarray(x[2:4]).reshape(2,)

        q_r = np.asarray(q_r).reshape(2,)
        q_r_dot = np.asarray(q_r_dot).reshape(2,)
        q_r_ddot = np.asarray(q_r_ddot).reshape(2,)


        v = q_r_ddot + self.K_d @ (q_r_dot - q_dot) + self.K_p @ (q_r - q)
        u = self.model.M(x) @ v + self.model.C(x) @ q_dot


        return u

