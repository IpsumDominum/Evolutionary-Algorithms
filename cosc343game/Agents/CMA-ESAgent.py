import numpy as np
import logging
import os
class game_stat:
    LEARNING_RATE = 0.01
    playerName = "myAgent"
    nPop = 34
    nPercepts = 75  #This is the number of percepts
    nActions = 7    #This is the number of actionss
    mutation_rate = 0.4
    num_elites = 10
    a_u = 0.1
    a_s = 0.1
    a_cp = 0.1
    a_c1 = 0.1
    a_clambda = 0.5
    d = 0.9
    t = 1
    mean = 0
    variance = 1
    C = np.identity(75)
    p_s = 0
    p_c = 0
class MyCreature:
    def __init__(self,mean=0,variance=1):
        netshape = np.array([
            (game_stat.nPercepts,game_stat.nActions),
        ])
        self.chromosomes = []
        for i in range(len(netshape)):
            self.chromosomes.append(np.random.normal(mean,variance,size=netshape[i]).astype(np.float32))
    def sample(self,mean,C):
        for i in range(75):
            self.chromosomes[0][i] = mean + np.random.normal(0,C[i][i])
        for i in range(75):
            self.chromosomes[0][i] = self.chromosomes[0][i] / sum(self.chromosomes[0][i])
    def AgentFunction(self, percepts):        
        #-------Propagate Network-------        
        squashed = percepts.flatten()
        x = np.matmul(squashed,self.chromosomes[0]).flatten()        
        return x
    def sigmoid(self,x):
        return 1/1+np.exp(-0.5*x)
def new_sample(mean,C):
    child = MyCreature()
    child.sample(mean,C)
    return child
def get_fitness(creature):
    fitness = (5*creature.enemy_eats + 5*creature.strawb_eats + 10*creature.size) *creature.alive + (5*creature.enemy_eats + 5*creature.strawb_eats + 10*creature.size) *creature.alive + 0.0001
    return fitness
def normalize_fitness(fitness):
    fitness = np.array(fitness)
    return fitness/sum(fitness)
def newGeneration(old_population):

    game_stat.a_clambda = min(1,game_stat.num_elites/np.power(game_stat.nPop,2))
    #------------Get Fitness--------------
    new_population = []
    fitness = [get_fitness(pop) for pop in old_population]
    avg_fitness = np.mean(np.array(fitness))
    norm_fitness = normalize_fitness(fitness)#fitness-np.mean(fitness)/(np.std(fitness)+1e-6)
    #-----------Write To Data-------------
    with open("ESfit.txt",'a') as file:
        file.write(str(np.array(fitness).mean())+"\n")    
    
    
    #-----------Get Elites-------------
    elites = np.random.choice(old_population,size=game_stat.num_elites,p=norm_fitness)
    elite_chromosomes = np.array([elite.chromosomes[0] for elite in elites])

    #-----------Compute Square root of C-
    B,_ = np.linalg.eig(game_stat.C)
    
    D = np.identity(75)
    for i in range(75):
        for j in range(75):
            if(i==j):
                D[i][j] = B[i]
            else:
                D[i][j] = 0
    sqrt_C = np.matmul(B.T,np.matmul(D,B))

    #----------Get avg new chromosome----------
    avg_ymul = np.zeros((75,75))
    for i in range(game_stat.num_elites):
        avg_ymul = avg_ymul + np.matmul(elite_chromosomes[i],elite_chromosomes[i].T)
    avg_ymul = avg_ymul/i
    #----------Get new mean and variance etc----
    new_mean = game_stat.mean + np.mean(elite_chromosomes-game_stat.mean)
    game_stat.p_s = (1-game_stat.a_s)*game_stat.p_s \
                    + np.sqrt(game_stat.a_s*(2-game_stat.a_s)*game_stat.num_elites)*sqrt_C\
                    *(new_mean - game_stat.mean)/game_stat.variance
    new_variance =  game_stat.variance*np.exp(game_stat.a_s*((abs(game_stat.p_s)/np.random.normal(0,1))-1)/game_stat.d)
    game_stat.p_c = (1-game_stat.a_cp)*game_stat.p_c + np.sqrt(game_stat.a_cp*(2-game_stat.a_cp)*game_stat.num_elites)\
                    *(new_mean - game_stat.mean)/game_stat.variance
    game_stat.C = (1-game_stat.a_clambda-game_stat.a_c1)*game_stat.C + game_stat.a_c1*game_stat.p_c*game_stat.p_c*np.identity(75)\
                    +game_stat.a_clambda*avg_ymul        
    #--------------Get New Population--------------
    for i in range(game_stat.nPop):        
        child = new_sample(new_mean,game_stat.C)
        new_population.append(child)
    game_stat.t +=1
    game_stat.mean = new_mean
    game_stat.variance = new_variance
    return (new_population, avg_fitness)
