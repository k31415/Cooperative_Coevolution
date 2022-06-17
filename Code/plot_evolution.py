import numpy as np
import matplotlib.pyplot as plt
from ground_robot import Ground_Robot
from aerial_robot import Aerial_Robot
from simulator import Simulator
import os
import neat

folder_name = '../Results/'
file_name = 'results_'



def save_single_plot(load_name, save_name):
    # load results
    results = np.load(load_name,allow_pickle=True)

    res = []
    for i in range(len(results.files)):
        res.append(results['arr_'+str(i)])

    max_avg, min_avg, avg_avg = res[:3]
    genome_ground_robot = res[3][0]
    genome_aerial_robot = res[4][0]
    avg_of_best_fitness = res[5]



    # plot fitness per generation
    plt.figure()
    max_var = min_var = avg_var = [0 for i in range(0,len(max_avg))]
    plt.plot( max_avg, c='b', label='max fitness')
    plt.plot( min_avg,  c='y', label='min fitness')
    plt.plot(avg_avg,  c='r', label='average fitness')
    plt.plot(avg_of_best_fitness, c='g', label='average of best fitness')

    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.xlim(0,len(max_avg))
    plt.ylim(0,6)
    plt.legend()
    name3 = save_name+'_'+str(round(10*np.amax(max_avg)))
    plt.savefig(name3)
    plt.close()
    return avg_of_best_fitness


def save_average_plot(save_name, list_of_avg):
    avg = np.average(list_of_avg, axis= 0)
    var = np.std(list_of_avg, axis = 0)
    plt.figure()
    plt.errorbar(np.arange(0,len(avg)/2,0.5),avg, yerr=var, label= "std dev per generation", color = 'c', zorder = 0) #alpha = 0.2)
    plt.plot(np.arange(0,len(avg)/2,0.5),avg, label= "average per generation", color = 'b', zorder = 1)

    avg_avg = [np.average(avg) for x in range(0,len(avg))]
    max_avg = [np.amax(avg) for x in range(0,len(avg))]
    plt.ylim(0,6)
    plt.xlim(0,len(avg)/2)
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.legend()

    plt.savefig(save_name)


if __name__ == '__main__':
    # init plot properties
    plt.rcParams.update({'font.size': 14})

    for j in ["fiftc", "riftc", "firtc", "rirtc", "fifto", "rifto"]:
        list_of_avg = []
        for i in range(0,6):

            load_name = folder_name+file_name+str(j)+'_'+str(i)+'.npz'
            save_name =  folder_name +"evolution/"+file_name+ str(j) + '_'+str(i)
            avg_of_best = save_single_plot(load_name, save_name)
            list_of_avg.append(avg_of_best)

        avg = np.average(list_of_avg, axis= 0)
        save_name_summary = folder_name+"evolution/"+file_name+str(j)+"_summary"
        save_average_plot(save_name_summary, list_of_avg)
