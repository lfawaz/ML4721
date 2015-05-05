import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics
from sklearn.cluster import KMeans

ratings = pd.read_csv("/home/lfawaz/Downloads/movies_csv/ratings.txt",header=None)
ratings_test = pd.read_csv("/home/lfawaz/Downloads/movies_csv/ratings_test.txt",header=None)

ratings = np.array(ratings)
y_ture_all  = np.array(ratings_test)
y_ture = (y_ture_all[:,2])
df_ratings = pd.DataFrame(ratings,columns=list('UMR'))

all_users = np.unique(ratings[:,0])
all_movies  = np.unique(ratings[:,1])

N1 = np.amax(all_users)
N2 = np.amax(all_movies)

lamda = 10
dim = 20
var = 0.25
I = np.identity(dim).astype('int')
m = np.zeros([1,dim])


M = df_ratings.pivot(index='U',columns='M',values='R')
M.insert(1325,1325,np.nan)
M.insert(1414,1414,np.nan)
M.insert(1577,1577,np.nan)
M.insert(1604,1604,np.nan)
M.insert(1637,1637,np.nan)
M.insert(1681,1681,np.nan)


def euclideanDistance(u,v,l):
	d = 0
	for x in range (l):
		d += pow((u[x] - v[x]),2)
	#return math.sqrt(d)
	return d



#return all moveies for a specific user
def omega_users(i):
	return np.array(M.ix[i].dropna().index)
#return all users for a specific movies
def omega_movies(j):
	return np.array(M[j].dropna().index)

def vector_summation(omega,matrix):
	summation = np.zeros([dim,dim])
	matrix = np.matrix(matrix)
	for i in (omega):
		summation += matrix[i-1].T*matrix[i-1]
	return summation

def U_scalar_summation(omega,matrix,i):
	summation = np.zeros([1,dim])
	matrix = np.matrix(matrix)
	#print omega.shape,omega
	for j in (omega):
		#print j
		#Mij = int(df_ratings["R"][(df_ratings["U"] == i) & (df_ratings["M"] == j)])
		M_ = ratings[(ratings[:,0] == i) & (ratings[:,1] == j)]
		Mij = int(M_[:,2])
		summation += Mij*matrix[j-1]
	return summation#print Mij

def V_scalar_summation(omega,matrix,j):
	summation = np.zeros([1,dim])
	matrix = np.matrix(matrix)
	#print omega.shape,omega
	for i in (omega):
		#Mij = int(df_ratings["R"][(df_ratings["U"] == i) & (df_ratings["M"] == j)])
		#print Mij
		M_ = ratings[(ratings[:,0] == i) & (ratings[:,1] == j)]
		Mij = int(M_[:,2])
		summation += Mij*matrix[i-1]
	return summation

def return_M_pred(U,V):
	constant = lamda*var*I
	#U = np.random.multivariate_normal(mean=np.zeros(dim), cov=(1./lamda)*I, size=N1)
	#V = np.random.multivariate_normal(mean=np.zeros(dim), cov=(1./lamda)*I, size=N2)

	M_pred = np.zeros([N1,N2])	

	
	for i in range(N1):
		i = i+1
		if i % 10 == 0:
			print i
		u_denom = np.matrix(constant + vector_summation(omega_users(i),V))
		u_num   = np.matrix(U_scalar_summation(omega_users(i),V,i))
		u = np.linalg.inv(u_denom)*u_num.T
		U[i-1] = u.T
	

	for j in range(N2):
		j = j+1
		if j % 10 == 0:
			print j
		v_denom = np.matrix(constant + vector_summation(omega_movies(j),U))
		v_num   = np.matrix(V_scalar_summation(omega_movies(j),U,j))
		v = np.linalg.inv(v_denom)*v_num.T
		V[j-1] = v.T

	
	U = np.matrix(U)
	V = np.matrix(V)	
	return U, V

def return_y_pred(M_pred): 
	M_pred = np.array(M_pred)
	y_pred = np.zeros([len(y_ture)])
	for k in range(len(y_ture)):
		i = y_ture_all[k][0] - 1
		j = y_ture_all[k][1] - 1
		if M_pred[i][j] <= 1:
			y_pred[k] = 1
		elif M_pred[i][j] > 5:
			y_pred[k] = 5
		else:
			y_pred[k] = round(M_pred[i][j])
	return y_pred

def calculate_log_likelihood(U,V):
	r = len(ratings)
	first_term = 0
	second_term = 0
	third_term = 0
	for x in range(r):
		i   = ratings[x][0] - 1
		j 	= ratings[x][1] - 1
		Mij = ratings[x][2]
		first_term += (1/(2*var)) * np.linalg.norm((Mij - U[i]*V[j].T),2)
	for i in range(N1):
		second_term += (lamda/2)*np.linalg.norm(U[i].T,2)
	for j in range(N2): 
		third_term += (lamda/2)*np.linalg.norm(V[j].T,2)
	log_likelihood = -1*first_term - second_term - third_term
	return log_likelihood


def main():
	itr = 100
	rmse = np.zeros([itr])
	log_likelihood = np.zeros([itr])
	U = np.random.multivariate_normal(mean=np.zeros(dim), cov=(1./lamda)*I, size=N1)
	V = np.random.multivariate_normal(mean=np.zeros(dim), cov=(1./lamda)*I, size=N2)

	#U = np.random.multivariate_normal(mean=np.zeros(dim), cov=(1./lamda)*I, size=N1)
	#V = np.random.multivariate_normal(mean=np.zeros(dim), cov=(1./lamda)*I, size=N2)
	#print calculate_log_likelihood(U,V)
	for x in range(itr):
		M_pred = np.zeros([N1,N2])
		log_likelihood[x] = calculate_log_likelihood(U,V)	
		print 'loop:',x
		U, V = return_M_pred(U, V)
		M_pred = U*V.T
		y_pred = return_y_pred(M_pred)
		rmse[x] = sqrt(mean_squared_error(y_ture, y_pred))
		
		print rmse
		print log_likelihood
		#kms = KMeans(n_clusters=5,n_init=30)
		#kms.fit(U)
		#centroids = kms.cluster_centers_
		#print centroids
		
	#np.savetxt("/home/lfawaz/Downloads/rmse_iter_%d.csv" %x, rmse, delimiter=",")
	np.savetxt("/home/lfawaz/Downloads/movies_results/rmse.csv", rmse, delimiter=",")
	np.savetxt("/home/lfawaz/Downloads/movies_results/log_likelihood.csv", log_likelihood, delimiter=",")
	np.savetxt("/home/lfawaz/Downloads/movies_results/M_pred.csv", M_pred, delimiter=",")
	np.savetxt("/home/lfawaz/Downloads/movies_results/U.csv", U, delimiter=",")
	np.savetxt("/home/lfawaz/Downloads/movies_results/V.csv", V, delimiter=",")
		
main()

