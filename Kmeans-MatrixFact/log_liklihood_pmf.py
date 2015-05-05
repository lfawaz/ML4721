import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt


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


#return all moveies for a specific user
def omega_users(i):
	i = i+1
	return np.array(M.ix[i].dropna().index - 1 )
#return all users for a specific movies
def omega_movies(j):
	j = j+1
	return np.array(M[j].dropna().index - 1)

def vector_summation(omega,matrix):
	summation = np.zeros([dim,dim])
	matrix = np.matrix(matrix)
	for i in (omega):
		summation += matrix[i].T*matrix[i]
	return summation

def Mij(i,j):
	M_array = np.array(M)
	return M_array[i][j]

def U_scalar_summation(omega,matrix,i):
	summation = np.zeros([1,dim])
	matrix = np.matrix(matrix)
	M_array = np.array(M.fillna(0))
	for j in (omega):
		summation += M_array[i][j]*matrix[j]
	return summation

def V_scalar_summation(omega,matrix,j):
	summation = np.zeros([1,dim])
	matrix = np.matrix(matrix)
	M_array = np.array(M.fillna(0))
	for i in (omega):
		summation += M_array[i][j]*matrix[i]
	return summation

def log_Ui(omega,U,V,i):
	log_Ui = np.zeros([1,dim])
	U = np.matrix(U)
	V = np.matrix(V)
	M_array = np.array(M.fillna(0))
	for j in (omega):
		log_Ui += (1/var)*(M_array[i][j] - U[i]*V[j].T)*V[j] - (lamda*U[i])
	return log_Ui


def log_Vj(omega,U,V,j):
	log_Vj = np.zeros([1,dim])
	U = np.matrix(U)
	V = np.matrix(V)
	M_array = np.array(M.fillna(0))
	for i in (omega):
		log_Vj += (1/var)*(M_array[i][j] - V[j]*U[i].T)*U[i] - (lamda*V[j])
	return log_Vi

def return_M_pred():
	constant = lamda*var*I
	U = np.zeros([N1,dim])
	V = np.random.multivariate_normal(m[0], I/float(lamda), N2)
	log_U = np.zeros([N1,dim])
	log_V = np.random.multivariate_normal(m[0], I/float(lamda), N2)

	M_pred = np.zeros([N1,N2])	
	log_M_pred = np.zeros([N1,N2])

	
	for i in range(N1):
		if i % 100 == 0:
			print i
		u_denom = np.matrix(constant + vector_summation(omega_users(i),V))
		u_num   = np.matrix(U_scalar_summation(omega_users(i),V,i))
		u = np.linalg.inv(u_denom)*u_num.T
		U[i] = u.T
		log_U[i] = log_Ui(omega_users(i),U,V,i)
			

	for j in range(N2):
		if j % 100 == 0:
			print j
		v_denom = np.matrix(constant + vector_summation(omega_movies(j),U))
		v_num   = np.matrix(V_scalar_summation(omega_movies(j),U,j))
		v = np.linalg.inv(v_denom)*v_num.T
		V[i] = v.T
		log_V[j] = log_Vj(omega_movies(j),U,V,j)
	
	U = np.matrix(U)
	V = np.matrix(V)
	log_U = np.matrix(log_U)
	log_V = np.matrix(log_V)

	M_pred = U*V.T
	log_M_pred = log_U*log_V.T	
	return M_pred, log_M_pred

def return_y_pred(M_pred): 
	M_pred = np.array(M_pred)
	y_pred = np.zeros([len(y_ture)])
	for k in range(len(y_ture)):
		i = y_ture_all[k][0] - 1
		j = y_ture_all[k][1] - 1
		if M_pred[i][j] <= 1:
			y_pred[k] = 1
		else:
			y_pred[k] = round(M_pred[i][j])
	return y_pred

def main():
	itr = 1
	rmse = np.zeros([itr])
	joint_log = np.zeros([itr])

	for x in range(itr):
		M_pred = np.zeros([N1,N2])	
		log_M_pred = np.zeros([N1,N2])	
		print 'loop:',x
		M_pred, log_M_pred = return_M_pred()
		y_pred = return_y_pred(M_pred)
		rmse[x] = sqrt(mean_squared_error(y_ture, y_pred))
		joint_log[x] = np.sum(log_M_pred)

	np.savetxt("/home/lfawaz/Downloads/rmse.csv", rmse, delimiter=",")
	np.savetxt("/home/lfawaz/Downloads/joint_log.csv", joint_log, delimiter=",")
	np.savetxt("/home/lfawaz/Downloads/M_pred.csv", M_pred, delimiter=",")
main()

