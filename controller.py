import numpy as np
import scipy.linalg
import control as ct
import controlpy as cp

class Controller():
    def __init__(self):
        self.e = 0
        self.th = 0
        self.ve = 0
        self.dt = 1/15.
        self.L = 4.24 # meters
        self.e_prev = 0
        self.th_prev = 0
        # num = [[[1.],[1.],[1.0]],[[1],[1],[1.0]],[[1],[1],[1.0]]]
        # den = [[[1,0.001],[1,0.001],[1,0.001]],[[1,0.001],[1,0.001],[1,0.001]],[[1,0.001],[1,0.001],[1,0.001]]]
        # sys1 = ct.tf(num, den)
        # print(sys1)
        # self.We = sys1
        # self.Wes = ct.ss(ct.minreal(self.We))
        # print(self.Wes)
        # self.ne, _ = self.Wes.A.shape

        # self.Wu = ct.tf([1,2],[1,10])
        # self.Wus = ct.ss(ct.minreal(self.Wu))
        # self.nu, _ = self.Wus.A.shape
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
        
        C = np.array([[1,0.,0,0,0],\
                        [0,0,1.0,0,0],\
                        [0,0,0,0,1]])
        D = np.array([[0.0,0.],\
                        [0,0],\
                        [0,0]])
        Q = np.array([[1,0.,0,0,0],\
                        [0,1,0,0,0],\
                        [0,0,1,0,0],\
                        [0,0,0,1,0],\
                        [0,0,0,0,1]])

        R = np.array([[20,0],\
                        [0,20]])
        state = np.array([[self.e],\
                        [self.ed],\
                        [self.th],\
                        [self.th_d],\
                        [self.ve]])
        
        self.np,_=A.shape;

        P = ct.ss(A,B,C,D)
        # print(self.Wes.B)
        # An = np.array([[A,np.zeros((self.np,self.ne)),np.zeros((self.np,self.nu))],\
        #                 [self.Wes.B[0][0]*C,self.Wes.A,np.zeros((self.ne,self.nu))],\
        #                 [np.zeros((self.nu,self.np)), np.zeros((self.nu,self.ne)), self.Wus.A]])
        
        # B1 = np.array([[B,np.zeros((self.np,1))],\
        #                 [np.zeros((self.ne,1)), self.Wes.B],\
        #                 [np.zeros((self.nu,1)),np.zeros((self.nu,1))]])
        
        # B2 = np.array([[B],[np.zeros((self.ne,1))],[self.Wus.B]])
        # C1 = np.array([[np.zeros((1,self.np)),self.Wes.C, np.zeros((1,self.nu))],\
        #                 [np.zeros((1,self.np)), np.zeros((1,self.ne)), self.Wus.C]])
        
        # C2 = np.array([[np.negative(C), np.zeros((1,self.ne)), np.zeros((1,self.nu))]])
        # D12 = np.hstack((np.zeros((1,self.nu)),self.Wus.D))
        # D21 = np.array([0,1])



        if controller_type == "LQR":
            input = self.dlqr(A,B,Q,R,state)
        if controller_type == "H2":
            Kc = cp.synthesis.controller_H2_state_feedback(A, B2, B2, C1, D12)
    
        self.e_prev = self.e
        self.th_prev = self.th

        if input[0,0] > 100.0:
            input[0,0] = 100.0
        
        if input[0,0] < -100.0:
            input[0,0] = -100.0

        steering = input[0,0]*1/100.0
        return[[input[1,0],-steering]]