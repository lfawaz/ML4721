import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import normalize




with open("/home/lfawaz/nyt_data.txt") as ins:
	nyt_data = []
	for line in ins:
		 nyt_data.append(line.split(','))
		 
		 #print [x.split(':') for x in elements[0]]
		 #split_elements = elements.strip.split(':')
		 #print split_elements
		 #nyt_data.append(dict(zip(elements[::2],elements[1::2])))

X = np.zeros([3012,8447])

for j in range(len(nyt_data)):
	for k in range(len(nyt_data[j])-1):
		x = nyt_data[j][k].split(':')
		word_idx = int(x[0]) - 1
		word_count = int(x[1])
		X[word_idx][j] = word_count

r = 25

W = np.random.uniform(0,1,(X.shape[0],r))
H = np.random.uniform(0,1,(r,X.shape[1]))
tiny = math.pow(10,-16)

itr = 200
plot_list = np.zeros(itr)

def calculate_objective(X,WH):
	diver = (X * np.log(WH) - WH) * -1	
	nans_index = np.isnan(diver)
	diver[nans_index] = 0
	return np.sum(diver)

def output_graph(plot_list):
	graph = plt
	graph.plot(plot_list)
	graph.title('object function per %d iteration' %itr)
	np.savetxt("/home/lfawaz/WH.csv", WH, delimiter=",")
	graph.show()


def Update_H(X,W,H):
	X_WH = np.divide(X,(np.dot(W,H) + tiny))
	Wik = normalize(W.T, norm='l1', axis=1)  
	Multiplier = np.dot(Wik,X_WH)
	return H*Multiplier

def Update_W(X,W,H):
	X_WH = np.divide(X,(np.dot(W,H) + tiny))
	Hkj = normalize(H.T, norm='l1', axis=0) 
	Multiplier = np.dot(X_WH,Hkj)
	return W*Multiplier


for x in range(itr):
	print x
	H = Update_H(X,W,H)
	W = Update_W(X,W,H)
	WH = W.dot(H)
	calculate_objective(X,WH)
	plot_list[x] = calculate_objective(X,WH)
	print np.sum(W,axis=0),np.sum(H,axis=1),plot_list[x]


np.savetxt("/home/lfawaz/Downloads/hw5text/W_part2.csv", W, delimiter=",")
np.savetxt("/home/lfawaz/Downloads/hw5text/H_part2.csv", H, delimiter=",")
np.savetxt("/home/lfawaz/Downloads/hw5text/WH_part2.csv", WH, delimiter=",")	

output_graph(plot_list)
