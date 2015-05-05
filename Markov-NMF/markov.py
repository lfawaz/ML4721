import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.sparse as sparse


scores = pd.read_csv("/home/lfawaz/cfb2014scores.csv",header=None)
with open("/home/lfawaz/legend.txt","r") as ins:
	teams_label = []
	for line in ins:
		teams_label.append(line)


scores_array = np.array(scores)

teams = np.unique(scores_array[:,0])

M = np.zeros([len(teams),len(teams)])


for i in range(len(scores_array)):
	team_i_1 = scores_array[i][0] - 1
	team_s_1 = float(scores_array[i][1])

	team_i_2 = scores_array[i][2] - 1
	team_s_2 = float(scores_array[i][3])


	if team_s_1 > team_s_2:
		M[team_i_1][team_i_1] += (1 + (team_s_1/(team_s_1 + team_s_2)))
		M[team_i_2][team_i_1] += (1 + (team_s_1/(team_s_1 + team_s_2)))
		M[team_i_2][team_i_2] += (0 + (team_s_2/(team_s_2 + team_s_1)))
		M[team_i_1][team_i_2] += (0 + (team_s_2/(team_s_2 + team_s_1)))
	else:
		M[team_i_1][team_i_1] += (0 + (team_s_1/(team_s_1 + team_s_2)))
		M[team_i_2][team_i_1] += (0 + (team_s_1/(team_s_1 + team_s_2)))
		M[team_i_2][team_i_2] += (1 + (team_s_2/(team_s_2 + team_s_1)))
		M[team_i_1][team_i_2] += (1 + (team_s_2/(team_s_2 + team_s_1)))

M = normalize(M,norm='l1',axis=1)	

eigen_value , eigen_vector = sparse.linalg.eigs(M.T,k=1)
u_1 = eigen_vector/np.sum(eigen_vector)

plot_list = np.zeros(1000)

def plot_formula(wt,u):
	return abs(np.sum(wt ** 2) - np.sum(u_1 ** 2))

w = np.zeros([len(teams)])
w += 1./len(teams)

for i in range(1000):
	w = w.dot(M)
	#plot_formula(wt=w,u=u_1)	
	plot_list[i] = plot_formula(wt=w,u=u_1)	

	if i == 9 or i == 99 or i == 199 or i == 999:
		print "Step Number:", i+1
		top_20 = w.argsort()[-20:]
		top_20 = top_20[::-1]
		for j in range(len(top_20)):
			team_index = top_20[j]
			Name = teams_label[team_index]
			print "Rank:", j + 1, Name, "Rank Value: ", w[team_index]
		print "____________________________________________________"
#print "\t"
#print "W(1000) - W(infinity) = ",plot_list[999]
#print "\t"
#graph = plt
#graph.plot(plot_list)
#graph.title('t between 1 to 1000')
#graph.show()

