import matplotlib.pyplot as plt
plt.style.use('classic')
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()


rng = np.random.RandomState(0)
#x = np.linspace(0, 10, 500)
x = range(65)
#y = np.cumsum(np.ones((500,0)), 0)
#y = np.zeros(shape=(500,4))

def normalize(fit):
    for i in range(20):
        fit[(i)*25:(i+1)*25] = np.mean(fit[i*25:(i+1)*25])
    return fit
with open("NEAT/cartpole.txt",'r') as file:
    lines = file.readlines()
fit = np.zeros((65))

for n,line in enumerate(lines):
    fit[n] = float(line.replace("\n",""))
#fit = fit-np.mean(fit)/np.std(fit)
#fit = normalize(fit)
plt.plot(x,fit,color="blue",label="NEAT Cartpole")


plt.legend(loc='upperleft')
#plt.legend('A', ncol=2, loc='upper left')
plt.show()