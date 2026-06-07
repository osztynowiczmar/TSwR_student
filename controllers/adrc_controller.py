import numpy as np
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from controllers.pd_controller import PDDecentralizedController
from models.manipulator_model import ManiuplatorModel

class ADRController(Controller):
    def __init__(self, Tp, params):

        self.Tp = Tp
        self.model = ManiuplatorModel(Tp)

        self.joint_controllers = []

        for param in params:
            self.joint_controllers.append(ADRCJointController(*param, Tp))
            #self.joint_controllers.append(PDDecentralizedController(*param, Tp))

    def calculate_b_hat(self, x):
        M_hat = self.model.M(x)

        b_hat_1 = 1.0 / M_hat[0, 0]
        b_hat_2 = 1.0 / M_hat[1, 1]

        return np.array([b_hat_1, b_hat_2])
    
    
    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        

        x = np.asarray(x).reshape(4,)

        b_hat = self.calculate_b_hat(x)
        
        for i, controller in enumerate(self.joint_controllers):
            controller.set_b(b_hat[i])

        u = []

        for i, controller in enumerate(self.joint_controllers):
            u.append(controller.calculate_control([x[i], x[i+2]], q_d[i], q_d_dot[i], q_d_ddot[i]))
        u = np.asarray(u).reshape(2,)

        return u

