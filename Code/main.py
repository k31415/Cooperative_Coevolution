#here I try to implement neat algorithm
import numpy as np
import matplotlib.pyplot as plt
from ground_robot import Ground_Robot
from aerial_robot import Aerial_Robot
from simulator import Simulator
import time
import os
import neat
from neat.six_util import iteritems, itervalues
import sys

starting_time = 0

num_evals_per_pair = 10
max_num_gen = 700
num_time_steps = 400 # two steps per second
best_of_evaluation = 25


map_x = 550
map_y = 350
token_per_area = 1
run_type = "fiftc"
amount_of_runs = 1 # serial not parallel

rand_init = False
rand_token = False
map_bounded = True


# evaluate pairs of ground and aerial robot
#
# type = 0: evaluate all ground robots with best of aerial robots
# type = 1: evaluate all aerial robots with best of ground robots
def eval_genomes(pop, ind, config_ground, config_aerial, type = 0):
    if type == 0:
        for genome_id, genome in pop:
            fit = []
            for i in range(0,num_evals_per_pair):
                fit.append(calc_fitness(genome,ind, config_ground, config_aerial))
            genome.fitness = np.average(fit)
    elif type == 1:
        for genome_id, genome in pop:
            fit = []
            for i in range(0,num_evals_per_pair):
                fit.append(calc_fitness(ind, genome, config_ground, config_aerial))
            genome.fitness = np.average(fit)

# calculate fitness of two robots
def calc_fitness(genome_ground_robot, genome_aerial_robot, config_ground, config_aerial, print_behaviour = False):
    #generate simulator
    simulator = Simulator(genome_ground_robot, genome_aerial_robot, config_ground, config_aerial, arena_size=(map_x,map_y), rand_init = rand_init, rand_token = rand_token, map_bounded = map_bounded, token_per_area = token_per_area)
    # run simulation
    for i in range(num_time_steps):
        f = simulator.step()
        if print_behaviour:
            print(simulator.get_behaviour_charac(num_time_steps))
        if f == 1:
            break
    return simulator.ground_robot.collected_token

# create population and run the evolution
def run(config_ground, config_aerial,index):

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config_ground, config_aerial)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))

    # Run for up to max_num_gen generations.
    # winner_g, winner_a: best ground and aerial genomes over all generations
    # best_pair: best pair of each generation
    # ma, mi, av: max, min, average fitness over all generations
    winner_g, winner_a, best_pair, ma, mi, av = p.run(eval_genomes, max_num_gen)

    # evaluate best pairs best_of_evaluation times
    # max_fitn, avg_fitn, min_fitn: max, min, average fitness of best pair evaluations per generation
    # best_avg_pair: best result pair of final evaluation
    print("\nEvaluate best pairs...\n")
    max_fitn, avg_fitn, min_fitn, best_avg_pair = evaluate_best_pairs(best_pair,config_ground, config_aerial)

    # Show output of the most fit genome against training data.
    #TODO: do best output
    print('\nOutput:')
    print('\nBest Fitness overall:')
    print(np.amax(ma))
    print('\nBest fitness of final evaluation:')
    print(np.amax(max_fitn))

    # TODO: clean up
    # save best, worst, average fitness of each generation , best pair of average fitness,
    # avg_fitn: list of average fitnesses per generation after final evaluation
    name = '../Results/results_'+str(run_type)+'_'+ str(index)+'.npz'
    np.savez(name, ma, mi, av,[best_avg_pair[0]],[best_avg_pair[1]],avg_fitn)
    # save list of best pairs for each generation
    name = '../Results/results_best_'+str(run_type)+'_'+ str(index)+'.npz'
    np.savez(name, best_pair)

    # save full population of last generation
    gg = []
    for g in itervalues(p.population_ground):
        gg.append(g)
    aa = []
    for a in itervalues(p.population_aerial):
        aa.append(a)
    gg = np.array(gg, dtype="object")
    aa = np.array(aa, dtype="object")
    name = '../Results/results_full_pop_'+str(run_type)+'_'+ str(index)+'.npz'
    np.savez(name, [best_avg_pair[0]],[best_avg_pair[1]],gg,aa)

    print("Test "+str(run_type)+'_'+ str(index))


# evaluate best pairs of each evaluation
def evaluate_best_pairs(best_pair, config_ground, config_aerial):
    max_fitn = []
    avg_fitn = []
    min_fitn = []
    best_avg_pair = []
    for r in range(0,len(best_pair)):
        if (r % 5 == 0):
            print(r)
        ground = best_pair[r][0]
        aerial = best_pair[r][1]
        temp_fitn = []
        for i in range(0,best_of_evaluation):
            temp_fitn.append(calc_fitness(ground, aerial, config_ground, config_aerial, print_behaviour= False))

        max_fitn.append(np.amax(temp_fitn))
        avg_fitn.append(np.average(temp_fitn))
        min_fitn.append(np.amin(temp_fitn))
        if np.amax(avg_fitn) == avg_fitn[-1]:
            best_avg_pair=best_pair[r]

    return max_fitn, avg_fitn, min_fitn, best_avg_pair

if __name__ == '__main__':
    starting_time= time.time()
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path_ground = os.path.join(local_dir, 'config-ground')
    config_path_aerial = os.path.join(local_dir, 'config-aerial')

    # Load configuration.
    config_ground = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path_ground)
    config_aerial = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      config_path_aerial)


    for i in range(0,amount_of_runs):
        run(config_ground, config_aerial,i)

        now = time.time() - starting_time
        print('Time elapsed in seconds: {:1.1f}, and in minutes: {:1.1f}'.format(now, now/60))
        print('Run number '+ str(i) + ' from '+ str(amount_of_runs)+ ' runs overall.')

    now = time.time() - starting_time
    print('Time elapsed in seconds: {:1.1f}, and in minutes: {:1.1f}'.format(now, now/60))
