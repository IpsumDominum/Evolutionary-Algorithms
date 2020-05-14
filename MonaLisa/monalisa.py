import os
import cv2
from collections import OrderedDict
import numpy as np
class GENEMONA:
    def __init__(self):
        self.POPSIZE = 100
        self.ELITENUM = 10
        self.MUTATIONRATE = 0.8
        self.MONASHAPE = (20,20,3)
        self.mona = cv2.imread("mona.jpeg")
        self.mona = cv2.resize(self.mona,(self.MONASHAPE[0],self.MONASHAPE[1]))/255
        self.population = [self.get_new_creature() for i in range(self.POPSIZE)]
    def crossover(self,elites):
        new_pop = []
        for i in range(self.POPSIZE-self.ELITENUM):
            result = np.ones(self.MONASHAPE)
            batch = np.random.choice(range(0,self.ELITENUM),(2),replace=False)
            sperm = elites[batch[0]]
            egg= elites[batch[1]]
            chromo_shape = egg.shape
            for i in range(chromo_shape[0]):
                for j in range(chromo_shape[1]):
                    for k in range(chromo_shape[2]):
                        if(np.random.rand(1)<0.5):
                            result[i][j][k] = egg[i][j][k]
                        else:
                            result[i][j][k] = sperm[i][j][k]
            new_pop.append(result)
        return new_pop
    def mutate(self,population):
        mutated = []
        chromo_shape = population[0].shape
        for item in population:            
            for i in range(chromo_shape[0]):
                for j in range(chromo_shape[1]):
                    for k in range(chromo_shape[2]):
                        item[i][j][k] += -0.05+np.random.rand(1)*0.1
            mutated.append(item)
        return mutated
    def perturb(self,best):
        offsprings = []
        chromo_shape = best.shape
        for i in range(self.POPSIZE-1):            
            item = best.copy()
            for i in range(chromo_shape[0]):
                for j in range(chromo_shape[1]):
                    for k in range(chromo_shape[2]):
                        item[i][j][k] += -0.05+np.random.rand(1)*0.1
            offsprings.append(item)
        return offsprings
    def get_elites(self,pop):
        best_fit = 0
        pop_fitness = {}
        elites = []
        best_elite = np.ones(self.MONASHAPE)
        for n,p in enumerate(pop):
            pop_fitness[n] = self.eval_fitness(p)
        top_n = self.get_top_n(pop_fitness,self.ELITENUM)        
        best = self.get_min(pop_fitness,{})
        for n,p in enumerate(pop):
            if(n in top_n.keys()):
                elites.append(p)
                if(n==best):
                    best_elite = p
        return elites,best_elite,pop_fitness[best]
    def get_top_n(self,dict,n,mode="min"):
        top_n = {}
        idx =0 
        if(mode=="min"):
            while(idx<n):
                minkey = self.get_min(dict,top_n)
                if(minkey==""):
                    pass
                else:
                    top_n[minkey] = dict[minkey]
                idx +=1                
        return top_n
    def get_min(self,dict,excludes):
        minitem = 100000
        minkey = ""
        for item in dict.keys():
            if(item not in excludes.keys()):                
                if dict[item]<=minitem:
                    minitem = dict[item]
                    minkey = item                
        return minkey
    def eval_fitness(self,creature):
        fitness = np.mean(np.power((self.mona - creature),2))
        return fitness
    def get_new_creature(self):
        creature = np.random.rand(self.MONASHAPE[0],self.MONASHAPE[1],self.MONASHAPE[2])    
        return creature    
    def add_pop(self,p1,p2):
        result = []
        for item in p1:
            result.append(item)
        for item in p2:
            result.append(item)
        return result
    def evolve(self):
        round = 0
        while(True):
            round +=1            
            elites,best,best_fit = self.get_elites(self.population)
            new_pop = self.crossover(elites)
            if (np.random.rand(1)<self.MUTATIONRATE):
                new_pop = self.mutate(new_pop)
            #new_pop = self.perturb(best)
            self.population = self.add_pop(elites,new_pop)
            cv2.imshow("best",cv2.resize(best,(512,512)))
            cv2.imshow("ref",cv2.resize(self.mona,(512,512)))
            print("EVO {} bestfit: {}".format(round,best_fit))
            k = cv2.waitKey(1)
            if(k==ord("q")):
                cv2.destroyAllWindows()
if __name__=="__main__":
    genemona = GENEMONA()
    genemona.evolve()
