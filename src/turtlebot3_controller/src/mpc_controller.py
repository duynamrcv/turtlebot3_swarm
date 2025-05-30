#! /usr/bin/env python3

import casadi as ca
import numpy as np

class MPC():
    def __init__(self, n_states, n_action, max_v, max_w, hoziron_length=20, time_step=0.1):
        self.opti, \
        self.opt_states, \
        self.opt_action, \
        self.opt_x, \
        self.opt_x_ref = self.setup_controller(n_states, n_action, max_v, max_w, hoziron_length, time_step)

    def setup_controller(self, n_states, n_action, max_v, max_w, N, dt):
        opti = ca.Opti()
        opt_states = opti.variable(n_states, N+1)
        opt_action = opti.variable(n_action, N)

        # Differential wheeled mobile robot kinematic
        f = lambda x_, u_: ca.vertcat(*[
            u_[0]*ca.cos(x_[2]),
            u_[0]*ca.sin(x_[2]),
            u_[1]
        ])

        # Initial condition
        opt_x_ref = opti.parameter(n_states, 1)
        opt_x = opti.parameter(n_states, 1)
        opti.subject_to(opt_states[:, 0] == opt_x)
        for i in range(N):
            x_next = opt_states[:,i] + f(opt_states[:,i], opt_action[:,i])*dt
            opti.subject_to(opt_states[:,i+1] == x_next)

        # Weighted squared error loss function
        q_cost = np.diag([10.0, 10.0, 1.0])
        r_cost = np.diag([0.5, 0.5])

        # Cost function
        obj = 0
        for i in range(N):
            state_error_ = opt_states[:,i] - opt_x_ref
            obj = obj + ca.mtimes([state_error_.T, q_cost, state_error_]) \
                      + ca.mtimes([opt_action[:,i].T, r_cost, opt_action[:,i]])
            if i != 0:
                diff_v = opt_action[0,i] - opt_action[0,i-1]
                obj += ca.mtimes([diff_v.T, r_cost[0,0], diff_v])
        opti.minimize(obj)

        opti.subject_to(opti.bounded(-max_v, opt_action[0,:], max_v))
        opti.subject_to(opti.bounded(-max_w, opt_action[1,:], max_w))

        opts_setting = {'ipopt.max_iter':2000,
                        'ipopt.print_level':0,
                        'print_time':0,}
                        # 'ipopt.acceptable_tol':1e-8,
                        # 'ipopt.acceptable_obj_change_tol':1e-6

        opti.solver('ipopt', opts_setting)
        return opti, opt_states, opt_action, opt_x, opt_x_ref
    
    def compute_action(self, pose, goal):
        # Set parameter
        self.opti.set_value(self.opt_x, pose)
        self.opti.set_value(self.opt_x_ref, goal)

        sol = self.opti.solve()

        action = sol.value(self.opt_action)[:,0]
        states = sol.value(self.opt_states)
        return action, states