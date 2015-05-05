import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import math


faces = pd.read_csv("/home/lfawaz/faces.csv",header=None)

X = np.array(faces)
r = 25

W = np.random.uniform(0,1,(X.shape[0],r))
H = np.random.uniform(0,1,(r,X.shape[1]))
tiny = math.pow(10,-16)

def calculate_objective(X,WH):
	return np.sum((X-WH) ** 2)

def output_graph(plot_list):
	graph = plt
	graph.plot(plot_list)
	graph.title('object function per %d iteration' %itr)
	graph.show()


def Update_H(X,W,H):
	WX = np.dot(W.T,X)
	WWH = np.dot(np.dot(W.T,W),H)
	Multiplier = WX/WWH
	return H*Multiplier

def Update_W(X,W,H):
	XH = np.dot(X,H.T)
	WHH = np.dot(W,np.dot(H,H.T))
	Multiplier = XH/WHH
	return W*Multiplier

itr = 200
plot_list = np.zeros(itr)

for x in range(itr):
	print x
	H = Update_H(X,W,H)
	W = Update_W(X,W,H)
	WH = W.dot(H)
	plot_list[x] = calculate_objective(X,WH)
	print plot_list[x]


np.savetxt("/home/lfawaz/Downloads/hw5text/W_part1.csv", W, delimiter=",")
np.savetxt("/home/lfawaz/Downloads/hw5text/H_part1.csv", H, delimiter=",")
np.savetxt("/home/lfawaz/Downloads/hw5text/WH_part1.csv", WH, delimiter=",")
output_graph(plot_list)	
