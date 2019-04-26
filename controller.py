import numpy as np
import scipy.linalg
import control as ct

class Controller():
    def __init__(self):
        self.e = 0
        self.th = 0
        self.ve = 0
        self.dt = 1/30.
        self.L = 4.24 # meters
        self.e_prev = 0
        self.th_prev = 0


    def proportional_vel(self):

        if self.th > 50.0:
            self.th = 50.0
        
        if self.th < -50.0:
            self.th = -50.0

        steering = self.th*1/50.0

        if self.ve > 2:
            self.ve = 2
        if self.ve < -2:
            self.ve = -2

        prop = -(1.0/2.0)*self.ve

        if prop < 0:
            brake = np.abs(prop)
            gas = 0
        elif prop > 0:
            gas = np.abs(prop)
            brake = 0
        else:
            gas, brake = 0, 0

        return [prop,steering]



    def dlqr(self,A,B,Q,R,state):
        X,L,K = ct.dare(A,B,Q,R)
        control = -np.matmul(K, state)
        return control

    def action(self,xte,angle_error,vel_error,vel,controller_type="LQR"):
        self.v = vel
        self.ve = vel_error
        self.e = xte
        self.th = angle_error
        self.ed = (self.e - self.e_prev)/self.dt
        self.th_d = (self.th - self.th_prev)/self.dt


        A = np.array([[1,self.dt,0,0,0],\
                        [0,0,self.v,0,0],\
                        [0,0,1,self.dt,0],\
                        [0,0,0,0,0],\
                        [0,0,0,0,1]])
        B = np.array([[0,0],\
                        [0,0],\
                        [0,0],\
                        [self.v/self.L,0],\
                        [0,self.dt]])

        Q = np.array([[80,0.,0,0,0],\
                        [0,1,0,0,0],\
                        [0,0,1,0,0],\
                        [0,0,0,1,0],\
                        [0,0,0,0,40]])

        R = np.array([[20,0],\
                        [0,20]])
        state = np.array([[self.e],\
                        [self.ed],\
                        [self.th],\
                        [self.th_d],\
                        [self.ve]])
        input = self.dlqr(A,B,Q,R,state)
        self.e_prev = self.e
        self.th_prev = self.th

        if input[0,0] > 100.0:
            input[0,0] = 100.0
        
        if input[0,0] < -100.0:
            input[0,0] = -100.0

        steering = input[0,0]*1/100.0
        return[[input[1,0],-steering]]