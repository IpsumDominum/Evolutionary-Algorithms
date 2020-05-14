import numpy as np
import logging
import os
class game_stat:
    NOISE_STD = 0.01
    LEARNING_RATE = 0.001
    GAME_STARTED = False
    playerName = "myAgent"
    nPop = 34
    nPercepts = 75  #This is the number of percepts
    nActions = 7    #This is the number of actionss
    num_elites = 20
class MyCreature:

    def __init__(self):
        netshape = np.array([
            (game_stat.nPercepts,20),
            (20,game_stat.nActions)
        ])
        self.chromosomes = []
        for i in range(len(netshape)):
            self.chromosomes.append(np.random.normal(size=netshape[i]).astype(np.float32))
        self.noise = np.array([
            np.zeros((game_stat.nPercepts,20)),
            np.zeros((20,game_stat.nActions)),
        ])
    def AgentFunction(self, percepts):        
        squashed = percepts.flatten()
        x = self.tanh(np.matmul(squashed,self.chromosomes[0]).flatten())
        for i in range(1,len(self.chromosomes)):
            x = self.tanh(np.matmul(x,self.chromosomes[i]).flatten())
        return x
    def apply_noise(self):        
        #Apply noise
        for i in range(len(self.chromosomes)):
            for j in range(len(self.chromosomes[i])):
                self.chromosomes[i][j] += game_stat.NOISE_STD *self.noise[i][j]
    def tanh(self,x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))+0.001
    def sigmoid(self,x):
        return 1/1+np.exp(-0.5*x)
def sample_noise(chromosomes):
    pos = []
    neg = []
    for i in range(len(chromosomes)):
        pos.append([])
        neg.append([])
        for j in range(chromosomes[i].shape[0]):                        
            noise_t = np.random.normal(size=chromosomes[i].shape[1]).astype(np.float32)
            pos[i].append(noise_t)
            neg[i].append(-noise_t)
    return np.array(pos), np.array(neg)
def get_fitness(creature):
    fitness = (5*creature.enemy_eats + 5*creature.strawb_eats + 10*creature.size) *creature.alive + (5*creature.enemy_eats + 5*creature.strawb_eats + 10*creature.size) *creature.alive + 0.0001
    return fitness
def normalize_fitness(fitness):
    fitness = np.array(fitness)
    return fitness/sum(fitness)
def update_weights_based_on_fitness(population,fitness,best_creature):
    new_pop = []
    weighted_noise = None
    fitness = normalize_fitness(fitness)
    for creature,fit in zip(population,fitness):
        if weighted_noise is None:
            weighted_noise =[]
            for i in range(len(creature.noise)):
                weighted_noise.append([])
                for j in range(len(creature.noise[i])):
                    weighted_noise[i].append([])
                    for k in range(len(creature.noise[i][j])):
                        weighted_noise[i][j].append(fit * creature.noise[i][j][k])
        else:
            for i in range(len(creature.noise)):
                for j in range(len(creature.noise[i])):
                    for k in range(len(creature.noise[i][j])):
                        weighted_noise[i][j][k] += fit * creature.noise[i][j][k]
    weighted_noise = np.array(weighted_noise)#conversion to numpy array
    #------------Update Best Creature-----------
    for i in range(len(creature.noise)):
        for j in range(len(creature.noise[i])):
            for k in range(len(creature.noise[i][j])):
                update = game_stat.LEARNING_RATE * (weighted_noise[i][j][k]/game_stat.nPop* game_stat.NOISE_STD)
                best_creature.chromosomes[i][j][k] += update                
    return best_creature
def get_best_creature(old_population):
    best_creature = MyCreature()
    best_fit = 0
    for pop in old_population:
        fit = get_fitness(pop)           
        if(fit >best_fit):
            best_fit = fit
            best_creature = pop 
    return best_creature
def newGeneration(old_population):
    #------------Get Fitness--------------
    
    new_population = []
    fitness = [get_fitness(pop) for pop in old_population]
    avg_fitness = np.mean(fitness)
    with open("ESfit.txt",'a') as file:
        file.write(str(np.array(fitness).mean())+"\n")            
    norm_fitness = fitness-np.mean(fitness)
    s = np.std(norm_fitness)
    if abs(s) > 1e-6:
        norm_fitness = norm_fitness/s
    p_fitness = normalize_fitness(fitness)
    elites = np.random.choice(old_population,replace=False,size=game_stat.num_elites,p=p_fitness)
    fitness = [get_fitness(pop) for pop in elites]
    #--------------Sample Noise--------------
    noise_batch = []
    neg_noise_batch = []
    for i in range(game_stat.nPop//2):
        noise,neg_noise = sample_noise(old_population[0].chromosomes)
        noise_batch.append(noise)
        neg_noise_batch.append(neg_noise)


    if game_stat.GAME_STARTED:
        #------------Repopulation------------
        best_creature = get_best_creature(elites)
        best_creature = update_weights_based_on_fitness(elites,norm_fitness,best_creature)
        avg_fitness = np.mean(fitness)
        game_stat.old_best = get_best_creature(elites)
    else:
        #------No Repop on First Round-------
        best_creature = get_best_creature(elites)
        avg_fitness = np.mean(fitness)
        game_stat.GAME_STARTED = True
    for i in range(game_stat.nPop):
        child = MyCreature()
        child.chromosomes = best_creature.chromosomes
        new_population.append(child)
    #-------------Add some noise-------------
    for i in range(game_stat.nPop//2-1):
        new_population[i].noise = noise_batch[i]
        new_population[i].apply_noise()
        new_population[i].isKing = False
    for i in range(game_stat.nPop//2,game_stat.nPop-2):
        new_population[i].noise = neg_noise_batch[i-game_stat.nPop//2]
        new_population[i].apply_noise()
        new_population[i].isKing = False
    return (new_population, avg_fitness)
