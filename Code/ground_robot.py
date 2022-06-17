import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import copy
import multiprocessing
import neat


# starting pose of the robot
init_pos = (40,40,np.pi/4)#(140,130,np.pi/2)


# Braitenberg vehicle
class Ground_Robot:
    def __init__(self, store_traj=False):
        self.x_pos = init_pos[0]
        self.y_pos = init_pos[1]
        self.theta = init_pos[2]
        # collection, token_detection and aerial_detection range
        self.c_range = 4
        self.t_range = 14
        self.a_range = 144

        self.start_pose = []

        # max speed and angular velocity
        self.v_max = 15/2
        self.w_max = (np.pi)/2
        self.v_min = 1/2

        # sensor input presence of token
        self.gi1 = 0
        self.gi2 = 0
        self.gi3 = 0
        self.gi4 = 0

        # amount of collected token
        self.collected_token = 0

        # sensor input presence of aerial robot
        self.ga1 = 0
        self.ga2 = 0
        self.ga3 = 0
        self.ga4 = 0

        self.store_traj = store_traj
        if store_traj:
            self.traj = ([],[])

    def setup(self, map_size,genome,config, rand_init=False):
        if rand_init:
            # start at random corner position with random orientation
            self.theta = np.random.uniform(0, 2*np.pi)
            self.x_pos = np.random.choice([40,510])
            self.y_pos = np.random.choice([40,310])
        else:
            # start at fixed position
            self.x_pos = init_pos[0]
            self.y_pos = init_pos[1]
            self.theta = init_pos[2]

        self.collected_token = 0
        # save starting position
        self.start_pose = [self.x_pos, self.y_pos, self.theta]
        # save net of robot
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)




    # execute robot behaviour, returns desired linear and angular velocities of the robot
    def update(self):

        # get wheel velocities from evolved behaviour
        v,w  = self.behav_evolved()
        v = v*self.v_max
        w = w*self.w_max

        # return linear and angular velocity to simulator for position calculation
        return v, w

    # return relative distance/angle to input pos
    def get_distance_and_angle(self, pos):

        dist = np.sqrt((self.x_pos-pos[0])**2 + (self.y_pos - pos[1])**2)
        angle = np.arccos((pos[1]-self.y_pos)/dist)

        if pos[0] < self.x_pos:
            angle = 2*np.pi - angle
        # angle in robot coordinates
        angle = angle - self.theta
        if angle < 0:
            angle = 2* np.pi + angle
        return dist, angle

    # get net output
    def behav_evolved(self):

        v, w = self.net.activate([self.gi1, self.gi2, self.gi3, self.gi4, self.ga1, self.ga2, self.ga3, self.ga4])
        return v, w

    # set position of robot manually
    def set_new_pos(self, px, py):

        # update position
        self.x_pos = px
        self.y_pos = py

        # save trajectory
        if self.store_traj:
            self.traj[0].append(px)
            self.traj[1].append(py)
