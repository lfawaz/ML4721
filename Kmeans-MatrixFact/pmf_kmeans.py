import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics
from sklearn.cluster import KMeans

U = pd.read_csv("/home/lfawaz/Downloads/movies_results/U.csv",header=None)
V = pd.read_csv("/home/lfawaz/Downloads/movies_results/V.csv",header=None)
U = np.array(U)
V = np.matrix(V)

with open("/home/lfawaz/Downloads/movies_csv/movies.txt","r") as ins:
	movies = []
	for line in ins:
		movies.append(line)

kms = KMeans(n_clusters=5,n_init=30)
kms.fit(U)
centroids = kms.cluster_centers_
dot_product = np.zeros([len(V)])

for i in range(len(centroids)):
	print "centroids",i
	for j in range(len(V)):
		dot_product[j] = centroids[i]*V[j].T
		closest_movies = dot_product.argsort()[-10:]
	for z in range(len(closest_movies)):
		movie_index = closest_movies[z]
		print movies[movie_index], "dot_product:",dot_product[movie_index]
	print '_______________________________________'
