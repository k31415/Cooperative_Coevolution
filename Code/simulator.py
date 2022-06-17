import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import copy
import multiprocessing
#from ann import ANN
from ground_robot import Ground_Robot
from aerial_robot import Aerial_Robot

# size of areas
area_size_x = 150
area_size_y = 150

class Simulator:
    # initialize simulator
    # genome_ground_robot, genome_aerial_robot
    # config_ground, config_aerial: config files of robots
    # arena_size: x and y size of the arena
    # rand_init: True = Robots will be initialized randomly
    # rand_token: True = Token will be placed randomly in area
    # map_bounded: True = bounded, False = unbounded map
    # token_per_area: how many token are located in each area
    # store_traj = True = Robots store their trajectories (usefull for plots)
    def __init__(self, genome_ground_robot, genome_aerial_robot, config_ground, config_aerial, arena_size=(550,350), rand_init = False, rand_token = False, map_bounded = False, token_per_area = 1,store_traj=False):

        self.map_size_x, self.map_size_y = arena_size
        self.timestep = 1
        self.token_per_area = token_per_area
        self.rand_init = rand_init
        self.bounded = map_bounded
        # initialize and setup ground robot
        self.ground_robot = Ground_Robot(store_traj=store_traj)
        self.ground_robot.setup((self.map_size_x, self.map_size_y),genome_ground_robot,config_ground, rand_init)
        # initialize and setup aerial robot
        self.aerial_robot = Aerial_Robot(store_traj=store_traj)
        self.aerial_robot.setup((self.map_size_x, self.map_size_y),genome_aerial_robot,config_aerial, rand_init)
        # set aerial robot position near to ground robot
        self.aerial_robot.set_new_pos(self.ground_robot.x_pos+1, self.ground_robot.y_pos+1)
        self.time_in_sensor_range_robots = 0
        self.avg_dist_in_sensor_range = []
        self.avg_dist_to_closest_item_ground = []
        self.avg_dist_to_closest_item_aerial = []

        if rand_token:
            self.token = []
            area_centers = [(100,90),(275,90),(450,90),(100,260),(275,260),(450,260)]

            for c in area_centers:
                for i in range(0,token_per_area):
                    self.token.append((np.random.uniform(c[0]-area_size_x/2,c[0]+area_size_x/2),np.random.uniform(c[1]-area_size_y/2,c[1]+area_size_y/2)))
        else:
            self.token = [(91,86), (91,261), (274,86), (274,261), (457,86), (457,261)]


    # simulate one time step for each robot
    def step(self):

        # update ground robot
        self.movement_ground_robot(self.ground_robot)
        dist_token_ground = self.token_sensors_ground_robot(self.ground_robot)
        self.ar_sensors_ground_robot(self.ground_robot,self.aerial_robot)
        # update aerial robot
        self.movement_aerial_robot(self.aerial_robot)
        dist_token_aerial = self.token_sensors_aerial_robot(self.aerial_robot, self.ground_robot)
        self.center_sensor_aerial_robot(self.aerial_robot)
        dist_robots = self.gr_sensors_aerial_robot(self.aerial_robot,self.ground_robot)

        # TODO: Needed?
        # calculate behaviour characteristics
        if dist_robots > 0:
            self.time_in_sensor_range_robots = self.time_in_sensor_range_robots +1
            self.avg_dist_in_sensor_range.append(dist_robots)
        else:
            self.avg_dist_in_sensor_range.append(2* self.aerial_robot.g_range)

        self.avg_dist_to_closest_item_ground.append(dist_token_ground)
        self.avg_dist_to_closest_item_aerial.append(dist_token_aerial)

        # break if all token collected
        if len(self.token) == 0:
            return 1

    # calculate and return behaviour characteristics
    def get_behaviour_charac(self, timesteps):
        # number of collected items
        a = self.ground_robot.collected_token

        # time of robots being in sensor range of each other (1 = always, 0 = never)
        b = self.time_in_sensor_range_robots/timesteps

        # average distance between robots
        max_dist = np.sqrt(self.map_size_x**2+ self.map_size_y**2)
        c = np.average(self.avg_dist_in_sensor_range)/max_dist

        # average distance of ground robot to nearest token
        d = np.average(self.avg_dist_to_closest_item_ground)/max_dist

        # average distance of aerial robot to nearest token
        e = np.average(self.avg_dist_to_closest_item_aerial)/max_dist
        return [a,b,c,d,e]

    # return position of ground robot
    def get_ground_robot_positions(self):
        return (self.ground_robot.x_pos, self.ground_robot.y_pos)

    # return position of aerial robot
    def get_aerial_robot_positions(self):
        return (self.aerial_robot.x_pos, self.aerial_robot.y_pos)

    # return list of all token positions
    def get_token_positions(self):
        return [x for (x,y) in self.token], [y for (x,y) in self.token]

    # move ground robot (one timestep)
    def movement_ground_robot(self, r):
        # get linear and angular velocity
        v,w = r.update()

        # new robot position
        x_pos_new = r.x_pos + v * self.timestep * np.cos(r.theta)
        y_pos_new = r.y_pos + v * self.timestep * np.sin(r.theta)

        # clip robot position into map boundaries
        if self.bounded:
            if x_pos_new > self.map_size_x:
                x_pos_new = self.map_size_x
            if x_pos_new < 0:
                x_pos_new = 0
            if x_pos_new > self.map_size_x:
                x_pos_new = self.map_size_x
            if x_pos_new < 0:
                x_pos_new = 0
            if y_pos_new > self.map_size_y:
                y_pos_new = self.map_size_y
            if y_pos_new < 0:
                y_pos_new = 0
            if y_pos_new > self.map_size_y:
                y_pos_new = self.map_size_y
            if y_pos_new < 0:
                y_pos_new = 0

        # update pose
        r.theta += w * self.timestep
        r.theta = r.theta % (2*np.pi)

        r.set_new_pos(x_pos_new, y_pos_new)

    # move aerial robot (one timestep)
    def movement_aerial_robot(self, a):
        x_pos, y_pos, rot = a.update()

        # new robot position
        x_pos_new = x_pos * self.timestep
        y_pos_new =  y_pos * self.timestep

        # clip robot position into map boundaries
        if self.bounded:
            if x_pos_new > self.map_size_x:
                x_pos_new = self.map_size_x
            if x_pos_new < 0:
                x_pos_new = 0
            if x_pos_new > self.map_size_x:
                x_pos_new = self.map_size_x
            if x_pos_new < 0:
                x_pos_new = 0
            if y_pos_new > self.map_size_y:
                y_pos_new = self.map_size_y
            if y_pos_new < 0:
                y_pos_new = 0
            if y_pos_new > self.map_size_y:
                y_pos_new = self.map_size_y
            if y_pos_new < 0:
                y_pos_new = 0

        # update pose
        a.theta += rot * self.timestep
        a.theta = a.theta % (2*np.pi)

        a.set_new_pos(x_pos_new, y_pos_new)

    # set token sensor values for ground robot
    def token_sensors_ground_robot(self,r):
        # check whether position of robot is near to token
        r.gi1 = 0
        r.gi2 = 0
        r.gi3 = 0
        r.gi4 = 0
        # set min_dist to maximum distance
        min_dist = np.sqrt(self.map_size_x**2+ self.map_size_y**2)
        # check distance to each token
        for t in self.token:
            dist, angle = r.get_distance_and_angle((t[0],t[1]))
            if dist < min_dist:
                min_dist = dist
            # collect token if token is in collection range
            if (dist < r.c_range):
                self.token.remove(t)
                r.collected_token = r.collected_token+1
            # set gi input values
            elif (dist < r.t_range):
                if angle < np.pi/4 or angle > 7*np.pi/4:
                    r.gi1 = 1
                elif angle < 3* np.pi/4:
                    r.gi2 = 1
                elif angle < 5* np.pi/4:
                    r.gi3 = 1
                elif angle < 7 * np.pi /4:
                    r.gi4 = 1
        return min_dist

    # set sensor values of ground robot to detect aerial robot
    def ar_sensors_ground_robot(self, r, a):
        r.ga1 = 0
        r.ga2 = 0
        r.ga3 = 0
        r.ga4 = 0
        # get the relative angle between robots
        dist, angle = r.get_distance_and_angle((a.x_pos,a.y_pos))
        if dist< r.a_range:
            if angle < np.pi/4 or angle > 7*np.pi/4:
                r.ga1 = 1
            elif angle < 3* np.pi/4:
                r.ga2 = 1
            elif angle < 5* np.pi/4:
                r.ga3 = 1
            elif angle < 7 * np.pi /4:
                r.ga4 = 1


    # set the token sensor values of the aerial robot
    def token_sensors_aerial_robot(self, a, r):
        a.ai1 = 0
        a.ai2 = 0
        a.ai3 = 0
        a.ai4 = 0
        a.ai5 = 0
        a.ai6 = 0
        min_dist = np.sqrt(self.map_size_x**2+ self.map_size_y**2)
        for t in self.token:
            dist, angle = a.get_distance_and_angle((t[0], t[1]))
            if dist < min_dist:
                min_dist = dist
            rel_dist = 1-dist/a.d_range
            if (dist < a.d_range):
                if angle < np.pi/6 or angle > 11* np.pi/6:
                    if rel_dist > a.ai1:
                        a.ai1 = rel_dist
                elif angle < 3*np.pi/6:
                    if rel_dist > a.ai2:
                        a.ai2 = rel_dist
                elif angle < 5*np.pi/6:
                    if rel_dist > a.ai3:
                        a.ai3 = rel_dist
                elif angle < 7*np.pi/6:
                    if rel_dist > a.ai4:
                        a.ai4 = rel_dist
                elif angle < 9*np.pi/6:
                    if rel_dist > a.ai5:
                        a.ai5 = rel_dist
                elif angle < 11*np.pi/6:
                    if rel_dist > a.ai6:
                        a.ai6 = rel_dist
        return min_dist

    # set sensor values of aerial robot to detect the ground robot
    def gr_sensors_aerial_robot(self, a, r):
        a.ag1 = 0
        a.ag2 = 0
        a.ag3 = 0
        a.ag4 = 0
        a.ag5 = 0
        a.ag6 = 0
        dist, angle = a.get_distance_and_angle((r.x_pos, r.y_pos))
        rel_dist = 1- dist/a.g_range
        if dist<a.g_range:
            if angle < np.pi/6 or angle > 11* np.pi/6:
                a.ag1 = rel_dist
            elif angle < 3*np.pi/6:
                a.ag2 = rel_dist
            elif angle < 5*np.pi/6:
                a.ag3 = rel_dist
            elif angle < 7*np.pi/6:
                a.ag4 = rel_dist
            elif angle < 9*np.pi/6:
                a.ag5 = rel_dist
            elif angle < 11*np.pi/6:
                a.ag6 = rel_dist
            return rel_dist
        # no ground robot in range
        return rel_dist


    # set sensor values of aerial robot to detect the center of map
    def center_sensor_aerial_robot(self, a):
        center_x = self.map_size_x/2
        center_y = self.map_size_y/2
        dist, angle = a.get_distance_and_angle((center_x, center_y))
        a.acd = dist/np.sqrt(center_x**2 + center_y**2)
        a.aca = angle
