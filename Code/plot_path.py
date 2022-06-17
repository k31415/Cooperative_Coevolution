# plot the trajectory of the best genome pair with a specified parameter set


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

init_pos = ((40,40,np.pi/2)) # starting pose of the robots
sim_steps = 400
folder_name = '../Results/'
file_name = 'results_'
token_per_area = 1

random_inits = False
rand_token = False
map_bounded = True

def dist_path(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def getSpeed(traj):
    len_path = []
    for i in range(1,len(traj[0])):
        len_path.append(np.abs(dist_path(traj[0][i], traj[1][i], traj[0][i-1], traj[1][i-1])))
    return len_path


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

    token = copy.deepcopy(simulator.token)
    for s in range(0,sim_steps):
        simulator.step()
    print("Finished simulator steps. Token collected: ")
    print(simulator.ground_robot.collected_token)



    # plot trajectories
    plt.figure()
    rect = plt.Rectangle([0,0],map_x,map_y, facecolor = "black", edgecolor = "black", alpha = 0.1)#, label = "map"
    plt.gca().add_patch(rect)
    plt.plot(ground_robots.traj[0], ground_robots.traj[1],label='ground robot', color = 'red', zorder = 2)
    plt.plot(aerial_robots.traj[0], aerial_robots.traj[1], label='aerial robot', color = 'blue', zorder = 1)



    simtok = simulator.token
    for tok in simtok:
        token.remove(tok)


    plt.scatter([x for (x,y) in token],[y for (x,y) in token], s=30, label = "collected token",color = 'black', marker = "o",facecolors = "none", zorder = 3)
    plt.scatter([x for (x,y) in simulator.token],[y for (x,y) in simulator.token], s=20, label = "uncollected token",marker = "o",color = 'black', zorder = 4)

    output_ground = getSpeed(ground_robots.traj)
    output_aerial = getSpeed(aerial_robots.traj)

    plt.axis('scaled')
    plt.xlim(-0.2*map_x,1.2*map_x)
    plt.ylim(-0.2*map_y,1.2*map_y)
    name= save_name_traj+"_"+str(simulator.ground_robot.collected_token)+"_"+"ri"+str(random_inits)+"_"+"rt"+str(rand_token)+"_"+"mb"+str(map_bounded)
    print(name)
    plt.savefig(name)
    return (output_ground, output_aerial)


if __name__ == '__main__':
    # init some plot properties
    plt.rcParams.update({'font.size': 16})
    local_dir = os.path.dirname(__file__)
    config_ground_file = os.path.join(local_dir, 'config-ground')
    config_aerial_file = os.path.join(local_dir, 'config-aerial')
    config_ground = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_ground_file)
    config_aerial = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      config_aerial_file)



    dict = {}
    for j in ["fiftc", "riftc", "firtc", "rirtc", "fifto", "rifto"]:
        avg_g = []
        max_g = None
        min_g = None

        avg_a = []
        max_a = None
        min_a = None

        for i in range(0,5):
            load_name = folder_name+file_name+str(j)+'_'+str(i)+'.npz'
            save_name_traj =  folder_name +"trajectories/"+file_name+ str(j) + '_'+str(i)
            gr, ar = save_plot(config_ground,config_aerial, load_name, save_name_traj)

            avg_g.append(np.average(gr))
            avg_a.append(np.average(ar))
            max = np.amax(gr)
            if max_g == None or max > max_g:
                max_g = max
            max = np.amax(ar)
            if max_a == None or max > max_a:
                max_a = max

            min = np.amin(gr)
            if min_g == None or min < min_g:
                min_g = min
            min = np.amin(ar)
            if min_a == None or min < min_a:
                min_a = min
        dict[j] = {'avg_g':avg_g, 'max_g': max_g, 'min_g':min_g,'avg_a':avg_a, 'max_a': max_a, 'min_a':min_a}



    for x in dict:
        avg_g = dict[x]['avg_g']
        avg_a = dict[x]['avg_a']
        max_g = dict[x]['max_g']
        max_a = dict[x]['max_a']
        min_g = dict[x]['min_g']
        min_a = dict[x]['min_a']
        print(str(round(np.average(avg_g)*100)/100)+ "&"+str(round(np.std(avg_g)*100)/100)+ "&"+str(round(max_g*100)/100)+ "&"+str(round(min_g*100)/100))
        print(str(round(np.average(avg_a)*100)/100)+ "&"+str(round(np.std(avg_a)*100)/100)+ "&"+str(round(max_a*100)/100)+ "&"+str(round(min_a*100)/100))
