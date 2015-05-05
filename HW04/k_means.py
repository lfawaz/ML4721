import random
import numpy as np
import math
from numpy.random import uniform
import matplotlib.pyplot as plt



def euclideanDistance(u,v,l):
	d = 0
	for x in range (l):
		d += pow((u[x] - v[x]),2)
	#return math.sqrt(d)
	return d

def create_clusters(assignments, observation, k_value, n):
	cluster_size = (assignments == k_value).sum()
	cluster 	 = np.zeros([cluster_size,2])
	cluster_counter = 0
	for i in range(n):
		if assignments[i] == k_value:
			cluster[cluster_counter] = observation[i]
			cluster_counter += 1
	return cluster

def cluster_mean(cluster):
	cluster_mean = np.array(cluster).astype('float').mean(axis=0)
	return cluster_mean

def k_means(observation,k):
	color = [ "red", "blue", "green", "yellow", "purple","orange"]
	number=random.randint(0,4)
	t = 20
	n = len(observation)
#intialize Mu1,Mu2,Mu3
	mu = random.sample(observation,k)
	assignments = np.zeros([n,1])
#optimize c
	mu_distances = np.zeros([n,k])
	min_distance = np.zeros([n,1])
	sum_distances = np.zeros([t,1])
	picked_colors = np.zeros([k,1])
	cgraph = plt


	for x in range(t):
		for i in range(n):
			for j in range(k):
				mu_distances[i][j] = euclideanDistance(observation[i],mu[j],2)
			min_distance[i] = np.amin(mu_distances[i],axis=0)
			assignments[i] = np.argmin(mu_distances[i],axis=0)

		for i in range(k):
			picked_colors[i] = number
			cluster_color = color[number]
			cluster = create_clusters(assignments,observation,i,n) 
			mu[i] = cluster_mean(cluster)
			cgraph.scatter(cluster[:,0],cluster[:,1],color=cluster_color)
			number=random.randint(0,5)
			while number in picked_colors:
				number = random.randint(0,5)

	
		sum_distances[x] = np.sum(min_distance)


	#graph = plt
	#graph.plot(sum_distances)
	#graph.title('Objective funtion for k = %d' % k)
	#graph.show()

	cgraph.title('clusters for k = %d' %k)
	cgraph.show()
def main():
	n = 500
	k = 6
	
	pi = np.zeros([n*3])
	for i in range(n*3):
		if i < n:
			pi[i] = 0.2
		if i >= n and i < n*2:
			pi[i] = 0.5
		if i >= n*2:
			pi[i] = 0.3

	m_1 = np.array([0,0])
	c_1 = np.array([[1,0],[0,1]])
	G1 = np.random.multivariate_normal(m_1, c_1, n)

	m_2 = np.array([3,0])
	c_2 = np.array([[1,0],[0,1]])
	G2 = np.random.multivariate_normal(m_2, c_2, n)

	m_3 = np.array([0,3])
	c_3 = np.array([[1,0],[0,1]])
	G3 = np.random.multivariate_normal(m_2, c_2, n)

	G = np.vstack([G1,G2,G3])

	observation = np.zeros([n,2])
        
	u = uniform(0, max(pi))

	for i in range(n):
		sample = pi>=u
		pick = random.sample(np.arange(np.sum(sample)), 1)        
		observation[i] = G[sample][pick]
		u = uniform(0, pi[sample][pick])
	for i in range(k):
		if i > 1:
			k_means(observation,i)

main()
