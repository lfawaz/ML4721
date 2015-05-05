import sys
import csv
import numpy as np
from scipy import stats
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt
import matplotlib.pyplot as plt

#import X csv file into a matrix
##X_reader = csv.reader(open("/home/lfawaz/Downloads/data_csv/X.txt","rb"),delimiter=',')
X_reader = csv.reader(open(str(sys.argv[1]),"rb"),delimiter=',')
X_list =list(X_reader)
X_matrix = np.array(X_list).astype('float')


#import Y csv file into a matrix
#y_reader = csv.reader(open("/home/lfawaz/Downloads/data_csv/y.txt","rb"),delimiter=',')
y_reader = csv.reader(open(str(sys.argv[2]),"rb"),delimiter=',')
y_list =list(y_reader)
y_matrix = np.array(y_list).astype('float')

#split data into training and testing data
X_train, X_test, y_train,y_test = train_test_split(X_matrix,y_matrix,test_size=20,train_size=372)

#create linear regression model
reg_model_train = linear_model.LinearRegression()

#fit training data
reg_model_train.fit(X_train,y_train)

#return coeffient
WML = np.array(reg_model_train.coef_).astype('float')

print('number of cylindars: ', WML.item(1))
print('displacement: ',WML.item(2))
print('horsepower: ' ,WML.item(3))
print('weight: ',WML.item(4))
print('acceleration: ',WML.item(5))
print('model year:',WML.item(6))




MAE_list = []

#Create loop to run test 1000 ties
for i in range(0,1000):

	#Split data
	X_train, X_test, y_train, y_test = train_test_split(X_matrix,y_matrix,test_size=20,train_size=372)
	#feed training data
	reg_model_train.fit(X_train,y_train)
	#predict testing data
	y_predict = reg_model_train.predict(X_test)
	#add mean absolute error to list
	MAE_list.append(mean_absolute_error(y_predict,y_test))

MAE_arr = np.array(MAE_list)

print('Mean of MAE after 1000 tests: ',np.mean(MAE_arr,axis=0))
print('STD of MAE after 1000 tests: ', np.std(MAE_arr,axis=0))

#Create loop for P equal 1,2,3,4
for n in range (1,5):
	RMSE_list = []
	#Create empty array
	error_array = np.zeros(shape=(1,1))

	#create loop for 1000 tests
	for i in range(0,1000):
		#define polynomail degree
		poly_reg_model = PolynomialFeatures(degree=n)
		X_train, X_test, y_train, y_test = train_test_split(X_matrix,y_matrix,test_size=20,train_size=372)
		#train data
		poly_reg_model.fit(X_train,y_train)
		#predict 
		y_predict = reg_model_train.predict(X_test)
		#calculate mean squared error
		RMSE_list.append(sqrt(mean_squared_error(y_predict,y_test)))
		RMSE_arr = np.array(RMSE_list)

		#subtract test and predict
		error = y_test - y_predict

		#add to array for poltting
		error_array = np.concatenate((error_array,error),axis=0)
		

	print('Mean of squared Error when P equals ',n,np.mean(RMSE_arr,axis=0))
	print('STD of squared Error when P equals ',n,np.std(RMSE_arr,axis=0))
		
		
	plt.hist(error_array,bins=500)
	plt.title('Error Distribution')
	plt.show()

	#calcualte maximum likelihood mean and variance
	print('Maximum Likelihood Mean when P equals ',n,np.mean(error_array,axis=0))
	print('Maximum Likelihood STD when P equals ',n,np.std(error_array,axis=0))
	






