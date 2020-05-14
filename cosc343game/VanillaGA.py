import numpy as np
import os
class game_stat:
    MUTATIONRATE = 0.6
    meta_gene = [0.1,0.1,1,10,1]
    best_fit = 0
    INITIALIZED = False
    crossover_mode = "uniform"
    count = 3
class MyCreature:
    def __init__(self):
        self.chromosomes = [
            #np.random.rand(75,4),
            #np.random.rand(75,1),
            np.random.normal(size=(1,10)),
            np.random.normal(size=(1,7)),
            np.random.normal(size=(75,10)),
            np.random.normal(size=(10,7)),
        ]
    def AgentFunction(self, percepts):
        out = np.zeros(7)
        squashed = percepts.flatten()
        x = self.tanh(np.matmul(squashed,self.chromosomes[2]).flatten() + self.chromosomes[0][0])
        x = self.tanh(np.matmul(x,self.chromosomes[3]).flatten() + self.chromosomes[1][0])
        #x = self.softmax(np.matmul(x,self.chromosomes[1]).flatten() + self.chromosomes[-1][0])
        #out[:4] = x[:4]
        #out[5] = x[4]
        #for i in range(3,len(self.chromosomes)):
        #    x = self.sigmoid(np.matmul(x,self.chromosomes[i]).flatten())
        out = x
        return out
    def save(self,i):
        with open("saved_penguin/{}".format(i),"w") as file:
            file.write(str(self.chromosomes)[7:])
    def tanh(self,x):
        return (np.exp(x) - np.exp(-x))/((np.exp(x) + np.exp(-x)))+0.001
    def sigmoid(self,x):
        return 1/1+np.exp(-0.5*x)
    def softmax(self,x):
        exp_sum = 0
        for n in x:
            exp_sum += np.exp(n)
        norm_x = [np.exp(n)/exp_sum for n in x]
        return norm_xp
def crossover(x1,x2):
    """
    Cross over each of the weights
    """
    result_chromos = [np.zeros((chromo.shape)) for chromo in x1.chromosomes]
    for i in range(len(x1.chromosomes)):
        for j in range(len(x1.chromosomes[i])):
            for k in range(len(x1.chromosomes[i][j])):
                if(np.random.rand(1)<0.5):
                    result_chromos[i][j][k] = x1.chromosomes[i][j][k]
                else:
                    result_chromos[i][j][k] = x2.chromosomes[i][j][k]
                if(np.random.rand(1)<game_stat.MUTATIONRATE):
                    result_chromos[i][j][k] += np.random.normal()
    return result_chromos
def normalize_fitness(fitness):
    fitness = np.array(fitness)
    return fitness/sum(fitness)
def newGeneration(old_population):
    if(not game_stat.INITIALIZED):
        game_stat.INITIALIZED = True
    new_population = list()
    fitness = np.zeros(len(old_population))
    fitness = [get_fitness_eval(creature) for creature in old_population]
    fitness = normalize_fitness(fitness)
    for n in range(len(old_population)):
        super_cute_baby = MyCreature()
        batch = np.random.choice(old_population,(2),replace=False,p=fitness)
        super_cute_baby.chromosomes = crossover(batch[0],batch[1])
        new_population.append(super_cute_baby)
    avg_fitness = np.array([unbiased_fitness(creature) for creature in old_population]).mean()
    return (new_population, avg_fitness)
def get_fitness_eval(creature):
    fitness = (5*creature.enemy_eats + 5*creature.strawb_eats + 10*creature.size) *creature.alive + (5*creature.enemy_eats + 5*creature.strawb_eats + 10*creature.size) *creature.alive + 0.0001
    return fitness
def unbiased_fitness(creature):
    f1 = 1
    f2 = 1
    f3 = 1
    f4 = 1
    f5 = 1
    fitness = (f1 *creature.turn + f2*creature.enemy_eats + f3*creature.strawb_eats + f4*creature.alive + f5*creature.size)
    return fitness