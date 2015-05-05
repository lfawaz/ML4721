import random
import numpy as np
import math
from numpy.random import uniform
import matplotlib.pyplot as plt
import pandas as pd


def euclideanDistance(u,v,l):
	d = 0
	for x in range (l):
		d += pow((u[x] - v[x]),2)
	#return math.sqrt(d)
	return d

V = pd.read_csv("/home/lfawaz/Downloads/movies_results/V.csv",header=None)
with open("/home/lfawaz/Downloads/movies_csv/movies.txt","r") as ins:
	movies = []
	for line in ins:
		movies.append(line)

V = np.array(V)

distances = np.zeros(len(V))

for i in range(3):
	print "movie",i+1
	for j in range(len(V)):
		distances[j] = euclideanDistance(V[i],V[j],20) 
	closest_movies = distances.argsort()[:6]
	#print closest_movies
	for z in range(len(closest_movies)):
		movie_index = closest_movies[z]
		print movies[movie_index], "distance:",distances[movie_index]
	print '____________________________'
