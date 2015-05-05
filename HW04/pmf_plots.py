import random
import numpy as np
import math
from numpy.random import uniform
import matplotlib.pyplot as plt
import pandas as pd
import csv


rmse = pd.read_csv("/home/lfawaz/Downloads/movies_results/rmse.csv",header=None)
log_likelihood = pd.read_csv("/home/lfawaz/Downloads/movies_results/log_likelihood.csv",header=None)
rmse = np.array(rmse)
log_likelihood = np.array(log_likelihood)

graph = plt
#graph.plot(rmse)
#graph.title('RMSE per interation')
graph.plot(log_likelihood)
graph.title('Log Likelihood per interation')
graph.show()
