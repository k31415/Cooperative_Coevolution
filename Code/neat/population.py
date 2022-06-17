"""Implements the core evolution algorithm."""
from __future__ import print_function

from neat.reporting import ReporterSet
from neat.math_util import mean
from neat.six_util import iteritems, itervalues
import numpy as np


class CompleteExtinctionException(Exception):
    pass


class Population(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config_ground, config_aerial, initial_state=None):
        self.reporters = ReporterSet()
        self.config_ground = config_ground
        self.config_aerial = config_aerial
        #set main config
        self.config = self.config_ground

        stagnation_ground = self.config_ground.stagnation_type(self.config_ground.stagnation_config, self.reporters)
        stagnation_aerial = self.config_aerial.stagnation_type(self.config_aerial.stagnation_config, self.reporters)

        self.reproduction_ground = self.config_ground.reproduction_type(self.config_ground.reproduction_config,
                                                     self.reporters,
                                                     stagnation_ground)
        self.reproduction_aerial = self.config_aerial.reproduction_type(self.config_aerial.reproduction_config,
                                                     self.reporters,
                                                     stagnation_aerial)
        if self.config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif self.config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif self.config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not self.config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(self.config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population_ground = self.reproduction_ground.create_new(config_ground.genome_type,
                                                           config_ground.genome_config,
                                                           config_ground.pop_size)
            self.species_ground = config_ground.species_set_type(config_ground.species_set_config, self.reporters)
            self.generation_ground = 0
            self.species_ground.speciate(config_ground, self.population_ground, self.generation_ground)

            self.population_aerial = self.reproduction_aerial.create_new(config_aerial.genome_type,
                                                           config_aerial.genome_config,
                                                           config_aerial.pop_size)
            self.species_aerial = config_aerial.species_set_type(config_aerial.species_set_config, self.reporters)
            self.generation_aerial = 0
            self.species_aerial.speciate(config_aerial, self.population_aerial, self.generation_aerial)
        else:
            raise RuntimeError("Not implemented")
            self.population, self.species, self.generation = initial_state


        self.best_genome_ground = self.population_ground.get(1)
        self.best_genome_aerial = self.population_aerial.get(1)
        self.best_genome_ground_overall = None
        self.best_genome_aerial_overall = None
        self.best_fitness_overall = 0

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def run(self, fitness_function, n=None):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        # list of tuples of best pair per generation
        best_pair = []
        # lists of max, min, average fitnesses per generation
        max_fitn = []
        min_fitn = []
        avg_fitn = []
        while n is None or k < n:
            k += 1
            self.reporters.start_generation(self.generation_ground)
            self.reporters.start_generation(self.generation_aerial)

            # Evaluate all genomes using the user-provided function.
            #### evaluate ground population
            fitness_function(list(iteritems(self.population_ground)), self.best_genome_aerial, self.config_ground, self.config_aerial, type = 0)

            best_ground = None
            fitness_ground = []
            fitness_per_gen = []
            best_fitness = 0

            # get best ground individual
            for g in itervalues(self.population_ground):
                fitness_per_gen.append(g.fitness)
                if best_ground is None or g.fitness > best_fitness:
                    best_ground = g
                    best_fitness = g.fitness
                fitness_ground.append(g.fitness)


            max_fitn.append(np.amax(fitness_per_gen))
            min_fitn.append(np.amin(fitness_per_gen))
            avg_fitn.append(np.average(fitness_per_gen))
            best_pair.append((best_ground, self.best_genome_aerial))

            # Track the best genomes ever seen.
            if self.best_genome_ground_overall is None or best_fitness > self.best_fitness_overall:
                self.best_genome_ground_overall = best_ground
                self.best_genome_aerial_overall = self.best_genome_aerial
                self.best_fitness_overall = best_fitness
            self.best_genome_ground = best_ground

            #### evaluate aerial population
            fitness_function(list(iteritems(self.population_aerial)), self.best_genome_ground, self.config_ground, self.config_aerial, type = 1)

            # Gather and report statistics.
            best_aerial = None
            best_fitness = 0
            fitness_aerial = []
            fitness_per_gen = []

            # get best aerial individual
            for a in itervalues(self.population_aerial):
                fitness_per_gen.append(a.fitness)
                if best_aerial is None or a.fitness > best_fitness:
                    best_aerial = a
                    best_fitness = a.fitness
                fitness_aerial.append(a.fitness)

            max_fitn.append(np.amax(fitness_per_gen))
            min_fitn.append(np.amin(fitness_per_gen))
            avg_fitn.append(np.average(fitness_per_gen))
            best_pair.append((self.best_genome_ground, best_aerial))

            print("\n reporter ground: ")
            self.reporters.post_evaluate(self.config_ground, self.population_ground, self.species_ground, best_ground)
            print("\n reporter aerial: ")
            self.reporters.post_evaluate(self.config_aerial, self.population_aerial, self.species_aerial, best_aerial)

            # Track the best genomes ever seen.
            if self.best_genome_ground_overall is None or best_fitness > self.best_fitness_overall:
                self.best_genome_aerial_overall = best_aerial
                self.best_genome_ground_overall= self.best_genome_ground
                self.best_fitness_overall = best_fitness
            self.best_genome_ground = best_ground
            self.best_genome_aerial = best_aerial


            # Create the next generation from the current generation.
            self.population_ground = self.reproduction_ground.reproduce(self.config_ground, self.species_ground,
                                                          self.config_ground.pop_size, self.generation_ground)
            self.population_aerial = self.reproduction_aerial.reproduce(self.config_aerial, self.species_aerial,
                                                          self.config_aerial.pop_size, self.generation_aerial)

            print("Best fitness of this generation: ")
            print(best_fitness)
            print("Best fitness overall: ")
            print(self.best_fitness_overall)

            # Divide the new population into species.
            self.species_ground.speciate(self.config_ground, self.population_ground, self.generation_ground)
            self.species_aerial.speciate(self.config_aerial, self.population_aerial, self.generation_aerial)

            self.reporters.end_generation(self.config_ground, self.population_ground, self.species_ground)
            self.reporters.end_generation(self.config_aerial, self.population_aerial, self.species_aerial)

            self.generation_ground += 1
            self.generation_aerial += 1

        #necessary?
        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.best_genome_ground_overall, self.best_genome_aerial_overall, best_pair, max_fitn, min_fitn, avg_fitn
