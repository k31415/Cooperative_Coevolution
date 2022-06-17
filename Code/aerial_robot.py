import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import copy
import multiprocessing
import neat

init_pos = (41,41,np.pi/4) # starting pose of the robots



# Braitenberg vehicle
class Aerial_Robot:
    def __init__(self, store_traj=False):
        self.x_pos = init_pos[0]
        self.y_pos = init_pos[1]
        self.theta = init_pos[2]

        self.start_pose = []

        # max speed and angular velocity
        self.v_max = 100/2
        self.w_max = (np.pi/2)/2
        self.v_min = 1/2

        # sensor inputs distance to closest item
        self.ai1 = 0
        self.ai2 = 0
        self.ai3 = 0
        self.ai4 = 0
        self.ai5 = 0
        self.ai6 = 0
        # sensor inputs distance to ground robot
        self.ag1 = 0
        self.ag2 = 0
        self.ag3 = 0
        self.ag4 = 0
        self.ag5 = 0
        self.ag6 = 0
        # range and relative angle to center of the arena
        self.acd = 0
        self.aca = 0
        # range to detect tokens
        self.d_range = 144
        # range to detect ground robot
        self.g_range = 144


        self.store_traj = store_traj
        if store_traj:
            self.traj = ([],[])

    def setup(self, map_size,genome, config, rand_init=False):
        if rand_init:
            # start at random position with random orientation
            self.x_pos = np.random.uniform(0, map_size[0])
            self.y_pos = np.random.uniform(0, map_size[1])
            self.theta = np.random.uniform(0, 2*np.pi)
        else:
            # start at fixed position
            self.x_pos = init_pos[0]
            self.y_pos = init_pos[1]
            self.theta = init_pos[2]


        # save starting position
        self.start_pose = [self.x_pos, self.y_pos, self.theta]

        self.net = neat.nn.FeedForwardNetwork.create(genome,config)


    # execute robot behaviour, returns desired linear and angular velocities of the robot
    def update(self):

        # get front-back velocity, sideways velocity and rotation velocity from evolved behaviour
        v_front, v_side, v_rotation = self.behav_evolved()

        v_front = v_front*self.v_max
        v_side = v_side *self.v_max
        v_rotation = v_rotation*self.w_max

        if v_front > self.v_max:
            v_front = self.v_max
        if v_front < -self.v_max:
            v_front = -self.v_max
        if v_side > self.v_max:
            v_side = self.v_max
        if v_side < -self.v_max:
            v_side = -self.v_max
        if v_rotation > self.w_max:
            v_rotation = self.w_max
        if v_rotation < -self.w_max:
            v_rotation = -self.w_max

        x_pos, y_pos = self.robot_to_world((v_front,v_side), self.theta)

        return x_pos, y_pos, v_rotation

    # transforms a point in robot coordinates to a point in world coordinates
    def robot_to_world(self, pos, ori):
        ct = np.cos(ori)
        st = np.sin(ori)
        return (self.x_pos + pos[0]*ct - pos[1]*st,
                self.y_pos + pos[1]*ct + pos[0]*st)

    # get relative distance/angle to input position
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
        v_front, v_side, v_rotation =  self.net.activate([self.ai1, self.ai2, self.ai3, self.ai4, self.ai5, self.ai6, self.ag1, self.ag2, self.ag3, self.ag4, self.ag5, self.ag6, self.acd, self.aca])

        return v_front, v_side, v_rotation

    # set robot position manually
    def set_new_pos(self, px, py):
        # update position
        self.x_pos = px
        self.y_pos = py

        # save trajectory
        if self.store_traj:
            self.traj[0].append(px)
            self.traj[1].append(py)
