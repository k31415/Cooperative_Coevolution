# save animation of one file


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import copy
import multiprocessing
from ground_robot import Ground_Robot
from aerial_robot import Aerial_Robot
from simulator import Simulator
import os
import neat

init_pos = (40,40,np.pi/4) # starting pose of the robots
sim_steps = 400
slowmo = 1 # 1 real time, 100 slow-motion
folder_name = '../Results/'
file_name = 'results_fiftc_2'
robots_moving = True
token_per_area = 1

random_inits = False
rand_token = False
map_bounded = True





def save_plot(config_ground,config_aerial, load_name, save_name_traj):
    # load results
    results = np.load(load_name,allow_pickle=True)

    res = []
    for i in range(len(results.files)):
        res.append(results['arr_'+str(i)])

    max_avg, min_avg, avg_avg = res[:3]
    genome_ground_robot = res[3][0]
    genome_aerial_robot = res[4][0]
    avg_of_best_fitness = res[5]


    # create robots, simulator, ...
    map_x = 550
    map_y = 350
    simulator = Simulator(genome_ground_robot, genome_aerial_robot, config_ground, config_aerial, arena_size=(map_x,map_y), rand_init = random_inits, rand_token = rand_token, map_bounded = map_bounded, token_per_area = token_per_area,store_traj=True)
    ground_robots = simulator.ground_robot
    aerial_robots = simulator.aerial_robot

    # setup animation
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    xtoken, ytoken = [], []
    ln1, = plt.plot([], [], 'ro')
    ln2, = plt.plot([], [], 'yo')
    ln3, = plt.plot([], [], 'bo')
    ln4, = plt.plot([], [], 'ro',markersize=ground_robots.a_range, linewidth=2, fillstyle = 'none')
    ln5, = plt.plot([], [], 'bo',markersize=aerial_robots.g_range, linewidth=2, fillstyle = 'none')
    ln6, = plt.plot([], [], 'ro', markersize=0, linewidth= 1, linestyle = '-')
    ln7, = plt.plot([], [], 'bo', markersize=0, linewidth= 1, linestyle = '-')
    ln8, = plt.plot([], [], 'ro',markersize=ground_robots.t_range, linewidth=2, fillstyle = 'none')

    # initialization function for animation
    def init():
        ax.set_xlim(0, map_x)
        ax.set_ylim(0, map_y)
        ax.set_aspect('equal')
        return ln1,ln2,ln3

    # animation step
    def sim(frame):

        global robots_moving
        if robots_moving:
            # update simulation
            xdata_ground_old, ydata_ground_old = simulator.get_ground_robot_positions()
            xdata_aerial_old, ydata_aerial_old = simulator.get_aerial_robot_positions()
            f = simulator.step()

            # get robot positions
            xdata_ground, ydata_ground = simulator.get_ground_robot_positions()
            xdata_aerial, ydata_aerial = simulator.get_aerial_robot_positions()
            xtoken, ytoken = simulator.get_token_positions()

            # plot positions
            if robots_moving == True:
                ln1.set_data(xdata_ground, ydata_ground)
                ln2.set_data(xtoken, ytoken)
                ln3.set_data(xdata_aerial, ydata_aerial)
                ln4.set_data(xdata_ground, ydata_ground)
                ln8.set_data(xdata_ground, ydata_ground)
                ln5.set_data(xdata_aerial, ydata_aerial)
                length_of_nose = 20
                x_pos_new = (length_of_nose+1)*xdata_ground - length_of_nose*xdata_ground_old
                y_pos_new = (length_of_nose+1)*ydata_ground - length_of_nose*ydata_ground_old
                ln6.set_data([xdata_ground,x_pos_new] , [ydata_ground, y_pos_new])
                x_pos_new_aerial = (length_of_nose+1)*xdata_aerial - length_of_nose*xdata_aerial_old
                y_pos_new_aerial = (length_of_nose+1)*ydata_aerial - length_of_nose*ydata_aerial_old
                ln7.set_data([xdata_aerial,x_pos_new_aerial] , [ydata_aerial, y_pos_new_aerial])

            if f == 1:
                robots_moving = False


        return ln1,ln2,ln3, ln4, ln5, ln6, ln7, ln8

    # run animation
    a = anim.FuncAnimation(fig, sim, interval=1*slowmo, frames=sim_steps, repeat=False, init_func=init, blit=True)
    #plt.show()

    writervideo = anim.FFMpegWriter(fps=60)
    a.save(save_name_traj+ '_animation.mp4', writer=writervideo)


    print("Amount of token collected: ")
    print(simulator.ground_robot.collected_token)

    # plot trajectories
    plt.figure()
    plt.scatter(ground_robots.traj[0], ground_robots.traj[1], s=2, label='ground robot', color = 'red')
    plt.scatter(aerial_robots.traj[0], aerial_robots.traj[1], s=2, label='aerial robot', color = 'blue')
    plt.scatter([x for (x,y) in simulator.token],[y for (x,y) in simulator.token], s=10, label = "token",color = 'orange')


    plt.axis('scaled')
    plt.xlim(-0.2*map_x,1.2*map_x)
    plt.ylim(-0.2*map_y,1.2*map_y)
    plt.legend()
    save_name_full= save_name_traj+"_"+str(simulator.ground_robot.collected_token)
    print(save_name_full)



if __name__ == '__main__':
    # init some plot properties
    plt.rcParams.update({'font.size': 14})
    local_dir = os.path.dirname(__file__)
    config_ground_file = os.path.join(local_dir, 'config-ground')
    config_aerial_file = os.path.join(local_dir, 'config-aerial')
    config_ground = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_ground_file)
    config_aerial = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      config_aerial_file)


    load_name = folder_name+file_name+'.npz'
    save_name_traj =  folder_name +"animation/"+file_name
    save_plot(config_ground,config_aerial, load_name, save_name_traj)
