from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        ### TODO implement ESO update
        if self.B.ndim == 1:
            q = float(np.asarray(q).squeeze())
            u = float(np.asarray(u).squeeze())
            y = q
            y_hat = ((self.W @ self.state).squeeze())
            error = y - y_hat
            state_dot = (self.A @ self.state + self.B.flatten() * u + self.L.flatten() * error)
            self.state = self.state + self.Tp * state_dot
        else:
        #Zadanie10
            q = np.asarray(q).reshape(-1)
            u = np.asarray(u).reshape(-1)
            y = q
            y_hat = self.W @ self.state
            error = y - y_hat
            state_dot = (self.A @ self.state + self.B @ u + self.L @ error)
            self.state = self.state + self.Tp * state_dot

    def get_state(self):
        return self.state
