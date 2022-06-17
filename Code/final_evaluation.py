# plot the trajectory of the best genome pair with a specified parameter set


import numpy as np
import matplotlib.pyplot as plt
import copy
from ground_robot import Ground_Robot
from aerial_robot import Aerial_Robot
from simulator import Simulator
import os
import neat

init_pos = ((40,40,np.pi/4)) # starting pose of the robots
sim_steps = 400 # two time steps per second
folder_name = '../Results/'
file_name = 'results_'
results_best_name = 'results_best_'
token_per_area = 1

random_inits = False
rand_token = False
map_bounded = True
amount_runs = 1 #50




def save_plot(config_ground,config_aerial, load_name, load_name_pairs, save_name):
    print("Start with: ")
    print(load_name)
    # load results
    results = np.load(load_name,allow_pickle=True)

    res = []
    for i in range(len(results.files)):
        res.append(results['arr_'+str(i)])

    max_avg, min_avg, avg_avg = res[:3]

    # load best pairs
    results_pairs = np.load(load_name_pairs, allow_pickle=True)

    res_pairs = []
    for i in range(len(results_pairs.files)):
        res_pairs.append(results_pairs['arr_'+str(i)])


    max_fitn = []
    avg_fitn = []
    min_fitn = []
    best_avg_pair = []

    for p in range(0,len(res_pairs[0])):
        if p % 50 == 0:
            print(p)
        genome_ground_robot_best = res_pairs[0][p][0]
        genome_aerial_robot_best = res_pairs[0][p][1]

        # create robots, simulator, ...
        map_x = 550
        map_y = 350
        temp_fitn = []
        for x in range(0,amount_runs):
            simulator = Simulator(genome_ground_robot_best, genome_aerial_robot_best, config_ground, config_aerial, arena_size=(map_x,map_y), rand_init = random_inits, rand_token = rand_token, map_bounded = map_bounded, token_per_area = token_per_area,store_traj=True)


            token = copy.deepcopy(simulator.token)
            for s in range(0,sim_steps):
                simulator.step()
            # add to list
            temp_fitn.append(simulator.ground_robot.collected_token)

        max_fitn.append(np.amax(temp_fitn))
        avg_fitn.append(np.average(temp_fitn))
        min_fitn.append(np.amin(temp_fitn))
        if np.amax(avg_fitn) == avg_fitn[-1]:
            best_avg_pair=[genome_ground_robot_best, genome_aerial_robot_best]


    # save the results as .npz file
    np.savez(save_name, max_avg, min_avg, avg_avg, [best_avg_pair[0]],[best_avg_pair[1]],avg_fitn)





if __name__ == '__main__':
    # init some plot properties
    plt.rcParams.update({'font.size': 7})
    local_dir = os.path.dirname(__file__)
    config_ground_file = os.path.join(local_dir, 'config-ground')
    config_aerial_file = os.path.join(local_dir, 'config-aerial')
    config_ground = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_ground_file)
    config_aerial = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      config_aerial_file)

    for j in ["fiftc"]:#, "riftc", "firtc", "rirtc", "fifto", "rifto"]:

        for i in range(0,6):

            load_name_pairs = folder_name+results_best_name+j+'_'+str(i)+'.npz'
            load_name = folder_name+file_name+str(j)+'_'+str(i)+'.npz'
            save_name =  folder_name +"new"+str(amount_runs)+"/"+file_name+ j + '_'+str(i)
            save_plot(config_ground,config_aerial,load_name, load_name_pairs, save_name)
