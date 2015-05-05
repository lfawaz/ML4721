import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import math
import operator
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

r_xtrain = csv.reader(open("/home/lfawaz/Downloads/mnist_csv/Xtrain.txt","rb"),delimiter=',')
r_xtest = csv.reader(open("/home/lfawaz/Downloads/mnist_csv/Xtest.txt","rb"),delimiter=',')
r_label_train = csv.reader(open("/home/lfawaz/Downloads/mnist_csv/label_train.txt","rb"),delimiter=',')
r_label_test = csv.reader(open("/home/lfawaz/Downloads/mnist_csv/label_test.txt","rb"),delimiter=',')
Q_images = csv.reader(open("/home/lfawaz/Downloads/mnist_csv/Q.txt","rb"),delimiter=',')


l_xtrain = list(r_xtrain)
l_xtest = list(r_xtest)

l_label_train = list(r_label_train)
l_label_test = list(r_label_test)

l_Q_images = list(Q_images)

for i in range(len(l_label_train)):
	l_label_train[i] = int(l_label_train[i][-1])

for i in range(len(l_label_test)):
	l_label_test[i] = int(l_label_test[i][-1])

xtrain = np.array(l_xtrain).astype('float')
xtest = np.array(l_xtest).astype('float')

m_Q_images = np.matrix(l_Q_images,float)
m_xtrain = np.matrix(l_xtrain,float)
m_xtest = np.matrix(l_xtest,float)

#y = np.array(m_Q_images*m_xtest[10].T).astype('float')
#plt.imshow(y.reshape(28,28),cmap="Greys")
#plt.show()


def euclideanDistance(u,v,l):
	d = 0
	for x in range (l):
		d += pow((u[x] - v[x]),2)
	#return math.sqrt(d)
	return d


def K_NearestNeighbor(test_set,k):
	distances = []
	for i in range(5000):
		dist = euclideanDistance(test_set,xtrain[i],20)
		distances.append((l_label_train[i],dist))
		distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors


def getprediction(neighbors):
	counts = [0,0,0,0,0,0,0,0,0,0]
	for x in range(len(neighbors)):
		prediction = int(neighbors[x])
		counts[prediction] += 1
	return counts.index(max(counts))

def K_NN_Classifier():
	for k in range(5):

		y_true = l_label_test
		y_pred = []
		y_error = []
		pred = []
		print "This is K: " , k+1
		for i in range(500):
			neighbors = K_NearestNeighbor(xtest[i],1) 
			pred = getprediction(neighbors)
			y_pred.append(pred)

			if y_pred[i] != y_true[i] and (k == 0 or k == 2 or k == 4):
				
				print "This is the prediction:",  y_pred[i]
				print "This is the true value:",  y_true[i]
				y_pred = np.array(m_Q_images*y_pred[i].T).astype('float')
				y_true = np.array(m_Q_images*y_true[i].T).astype('float')
				plt_pred.imshow(y_pred.reshape(28,28),cmap="Greys")
				plt_show.imshow(y_true.reshape(28,28),cmap="Greys")
				plt_pred.show()
				plt_show.show()
	
				
		
		conf_matrix = confusion_matrix(y_true,y_pred)
		
					
		print conf_matrix

def calculate_mean(class_list):

	l = list(class_list)
	mean_vector = np.array(l).astype('float').mean(axis=0)
	return mean_vector


def calculate_cov(class_list):

	l = list(class_list)
	matrix = np.matrix(l,float)

	mean_array = ([calculate_mean(l),]*len(l))
	mean_matrix = np.matrix(mean_array,float)
	
	matrix_vector = matrix - mean_matrix
	cov_matrix = (matrix_vector.T*matrix_vector)
	cov_matrix = (cov_matrix)/500
	return cov_matrix

def calculate_probability(data,mean,covariance):
	data = np.matrix(data,float)
	mean = np.matrix(mean,float)
	d_m = data - mean

	covariance = np.matrix(covariance,float)		
	exp = np.dot(np.dot(d_m,np.linalg.inv(covariance)),d_m.T)
	P = np.log(np.linalg.det(covariance)) + exp
	return P

def Bayes_Classifer():
	l_0 = []
	l_1 = []
	l_2 = []
	l_3 = []
	l_4 = []
	l_5 = []
	l_6 = []
	l_7 = []
	l_8 = []
	l_9 = []
	
	for i in range(l_label_train.index(0),l_label_train.index(1)):
		l_0.append(l_xtrain[i])

	for i in range(l_label_train.index(1),l_label_train.index(2)):
		l_1.append(l_xtrain[i])

	for i in range(l_label_train.index(2),l_label_train.index(3)):
		l_2.append(l_xtrain[i])

	for i in range(l_label_train.index(3),l_label_train.index(4)):
		l_3.append(l_xtrain[i])

	for i in range(l_label_train.index(4),l_label_train.index(5)):
		l_4.append(l_xtrain[i])

	for i in range(l_label_train.index(5),l_label_train.index(6)):
		l_5.append(l_xtrain[i])

	for i in range(l_label_train.index(6),l_label_train.index(7)):
		l_6.append(l_xtrain[i])


	for i in range(l_label_train.index(7),l_label_train.index(8)):
		l_7.append(l_xtrain[i])

	for i in range(l_label_train.index(8),l_label_train.index(9)):
		l_8.append(l_xtrain[i])	

	for i in range(l_label_train.index(9),5000):
		l_9.append(l_xtrain[i])

	mean_array_l0 = calculate_mean(l_0)
	cov_matrix_l0 = calculate_cov(l_0)

	mean_matrix_l0 = np.matrix(mean_array_l0,float)
	y = np.array(m_Q_images*mean_matrix_l0.T).astype('float')
	plt.imshow(y.reshape(28,28),cmap="Greys")
	plt.show()
	
	mean_array_l1 = calculate_mean(l_1)
	cov_matrix_l1 = calculate_cov(l_1)

	mean_matrix_l1 = np.matrix(mean_array_l1,float)
	y = np.array(m_Q_images*mean_matrix_l1.T).astype('float')
	plt.imshow(y.reshape(28,28),cmap="Greys")
	plt.show()
	
	
	mean_array_l2 = calculate_mean(l_2)
	cov_matrix_l2 = calculate_cov(l_2)
	
	mean_matrix_l2 = np.matrix(mean_array_l2,float)
	y = np.array(m_Q_images*mean_matrix_l2.T).astype('float')
	plt.imshow(y.reshape(28,28),cmap="Greys")
	plt.show()
	
	mean_array_l3 = calculate_mean(l_3)
	cov_matrix_l3 = calculate_cov(l_3)
	
	mean_matrix_l3 = np.matrix(mean_array_l3,float)
	y = np.array(m_Q_images*mean_matrix_l3.T).astype('float')
	plt.imshow(y.reshape(28,28),cmap="Greys")
	plt.show()
	
	mean_array_l4 = calculate_mean(l_4)
	cov_matrix_l4 = calculate_cov(l_4)
	
	mean_matrix_l4 = np.matrix(mean_array_l4,float)
	y = np.array(m_Q_images*mean_matrix_l4.T).astype('float')
	plt.imshow(y.reshape(28,28),cmap="Greys")
	plt.show()
	
	mean_array_l5 = calculate_mean(l_5)
	cov_matrix_l5 = calculate_cov(l_5)

	mean_matrix_l5 = np.matrix(mean_array_l5,float)
	y = np.array(m_Q_images*mean_matrix_l5.T).astype('float')
	plt.imshow(y.reshape(28,28),cmap="Greys")
	plt.show()
	
	mean_array_l6 = calculate_mean(l_6)
	cov_matrix_l6 = calculate_cov(l_6)

	mean_matrix_l6 = np.matrix(mean_array_l6,float)
	y = np.array(m_Q_images*mean_matrix_l6.T).astype('float')
	plt.imshow(y.reshape(28,28),cmap="Greys")
	plt.show()
	
	mean_array_l7 = calculate_mean(l_7)
	cov_matrix_l7 = calculate_cov(l_7)

	mean_matrix_l7 = np.matrix(mean_array_l7,float)
	y = np.array(m_Q_images*mean_matrix_l7.T).astype('float')
	plt.imshow(y.reshape(28,28),cmap="Greys")
	plt.show()
	
	mean_array_l8 = calculate_mean(l_8)
	cov_matrix_l8 = calculate_cov(l_8)

	mean_matrix_l8 = np.matrix(mean_array_l8,float)
	y = np.array(m_Q_images*mean_matrix_l8.T).astype('float')
	plt.imshow(y.reshape(28,28),cmap="Greys")
	plt.show()
	
	mean_array_l9 = calculate_mean(l_9)
	cov_matrix_l9 = calculate_cov(l_9)

	mean_matrix_l9 = np.matrix(mean_array_l9,float)
	y = np.array(m_Q_images*mean_matrix_l9.T).astype('float')
	plt.imshow(y.reshape(28,28),cmap="Greys")
	plt.show()
	
	
	
	#print calculate_cov(l_0)

	#print calculate_probability(xtest[1],mean_array_l0,cov_matrix_l0)]

	y_pred = []
	y_true = l_label_test
	for i in range(500):
		l = []
		l.append(calculate_probability(xtest[i],mean_array_l0,cov_matrix_l0))
		l.append(calculate_probability(xtest[i],mean_array_l1,cov_matrix_l1))
		l.append(calculate_probability(xtest[i],mean_array_l2,cov_matrix_l2))
		l.append(calculate_probability(xtest[i],mean_array_l3,cov_matrix_l3))
		l.append(calculate_probability(xtest[i],mean_array_l4,cov_matrix_l4))
		l.append(calculate_probability(xtest[i],mean_array_l5,cov_matrix_l5))
		l.append(calculate_probability(xtest[i],mean_array_l6,cov_matrix_l6))
		l.append(calculate_probability(xtest[i],mean_array_l7,cov_matrix_l7))
		l.append(calculate_probability(xtest[i],mean_array_l7,cov_matrix_l8))
		l.append(calculate_probability(xtest[i],mean_array_l7,cov_matrix_l9))

		y_pred.append(l.index(max(l)))

	conf_matrix = confusion_matrix(y_true,y_pred)
	print conf_matrix


def sigmoid(X):
	f = 1/(1.0 + np.exp(-0.1 * X))
	return f

def softmax(W,x,k):
	divisor = 0
	w = np.matrix(W)
	vector = np.exp(np.dot(w[k],x.T))
	for i in range(10):
		if i != k:
			divisor += np.exp(np.dot(W[k],x.T))

	result = vector/divisor
	return result

def get_mn_probability(W,x):
	l = []
	w = np.matrix(W)
	x = np.matrix(x)
	l = w*x.T
	return l
	
def softmax_Classififer():
	y_true = l_label_test
	w = np.zeros((10,20))
	y_matrix = np.zeros((5000,10))
	for i in range(len(xtrain)):
		j = float(l_label_train[i])
		y_matrix[i][j] = 1


	for i in range(1000):
		print "Started Train Round: ", i
		for j in range(len(xtrain)):
			for k in range(10):
				w[k] = w[k] + (0.1/5000)*(y_matrix[j][k] - softmax(w,xtrain[j],k))*xtrain[j]
	y_pred = []	
	dist = []
	for i in range(500):
		print "Started Test Round: ", i
		l = []
		l = get_mn_probability(w,xtest[i]).tolist()
		dist.append(max(l))
		y_pred.append(l.index(max(l)))

	conf_matrix = confusion_matrix(y_true,y_pred)
	plt.plot(dist)
	plt.show()

		
	print conf_matrix
def Main():
	K_NN_Classifier()
	Bayes_Classifer()
	softmax_Classififer()
	
	
Main()