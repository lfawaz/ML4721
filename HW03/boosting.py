import numpy as np 
import random
from numpy.random import uniform
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
import math
from sklearn.metrics import confusion_matrix


def bootstrap_sampler(w, n):

    b = np.zeros(n, dtype=np.int)
    k = np.arange(len(w))
    w = np.array(w) 
    u = uniform(0, max(w))

    for i in range(n):
        sample = w>=u
        pick = random.sample(np.arange(np.sum(sample)), 1)        
        b[i] = k[sample][pick]
        u = uniform(0, w[sample][pick])

    return b
    
def bootstrap_plot():
	w = np.array([0.1,0.2,0.3,0.4])
	for n in (100,200,300,400,500):
		x = bootstrap_sampler(w,n)
		plt.plot(x)
		plt.show()

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
    cov_matrix /= 500
    return cov_matrix

def calculate_pi(class_list,label):
    l=list(class_list)
    pi = float(l.count(label))/float(len(class_list))
    return pi

def calculate_w(m_1,m_0,cov):
    return np.linalg.inv(cov).dot(m_1 - m_0)

def calculate_w0(pi_1,pi_0,m_1,m_0,cov):
 #   cov_1 = np.delete(cov,0,axis=0)
 #   cov = np.delete(cov_1,0,axis=1)
 #   m_1 = np.delete(m_1,0,axis=0)
 #   m_0 = np.delete(m_0,0,axis=0)

    return (np.log(pi_1/pi_0) - (0.5*((m_1 + m_0).T.dot(np.linalg.inv(cov)).dot(m_1-m_0))))

def bayes(X_train, y_train):

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_train = np.delete(X_train,0,axis=1)

    X_cancer = []
    X_nocancer = []

    #separate classes to calculate mean and pi 
    for i in range(0,500):
        if y_train[i] == 1:
            X_cancer.append(X_train[i])   
        else:
            X_nocancer.append(X_train[i])

    a_X_cancer = np.array(X_cancer)
    a_X_nocancer = np.array(X_nocancer)
    a_X_all = np.array(X_train)

    m_1     = calculate_mean(a_X_cancer)
    m_0     = calculate_mean(a_X_nocancer)
    cov     = calculate_cov(a_X_all)
    pi_1    = calculate_pi(y_train,1)
    pi_0    = calculate_pi(y_train,-1)
    w0      = calculate_w0(pi_1,pi_0,m_1,m_0,cov)
    w       = calculate_w(m_1,m_0,cov)  
    w       = np.column_stack((w0,w))

    return w

def return_bayes_noboosting(X_train, X_test, y_train, y_test):

    X_train = np.array(X_train)
    X_test  = np.array(X_test)
    y_train = np.array(y_train)
    y_test  = np.array(y_test)

    w = bayes(X_train, y_train)

    y_pred  = []
    for i in range(len(X_test)):
       if float((X_test[i]*w.T)) >= 0:
            y_pred.append(1)
       else:
            y_pred.append(-1)   

    conf_matrix = confusion_matrix(y_test,y_pred)

    trace = sum(conf_matrix[i][i] for i in range(conf_matrix.shape[0]))

    print 'accurray nobootstrap:', trace/float(len(y_test))

def bayes_boosting(X_train, X_test, y_train, y_test):

    X_train = np.array(X_train)
    X_test  = np.array(X_test)
    y_train = np.array(y_train)
    y_test  = np.array(y_test)

    wt = np.empty(len(X_train)).astype('float')
    i = 1/float(len(X_train))
    wt.fill(i)    
    n = len(X_train)
    j = len(y_test)
    y_pred_test  =   np.zeros([j,1], dtype=int)
    y_pred_train =  np.zeros([n,1], dtype=int)
    error_array =   np.zeros([1000,2])
    alpha_epsilon = np.zeros([1000,2])
    dp_w_trace       = np.zeros([1000,3])


    for t in range(1000):
        print "this is loop number: ", t
        bootstrap_X = np.zeros([n,10])
        bootstrap_y = np.zeros([n,1], dtype=int)
        u = uniform(0, max(wt))

        for i in range(n):
            sample = wt>=u
            pick = random.sample(np.arange(np.sum(sample)), 1)        

            bootstrap_X[i] = X_train[sample][pick]
            bootstrap_y[i] = y_train[sample][pick]
            u = uniform(0, wt[sample][pick])

        w = bayes(bootstrap_X, bootstrap_y)  

       

        for i in range(n):
            if float((X_train[i]*w.T)) >= 0:
                y_pred_train[i] += 1
            else:
                y_pred_train[i] += -1   

        y_error = y_pred_train * y_train
    
        epsilon = 0
        error_train_count = 0
        for i in range(len(y_error)):
            if y_error[i] < 0:
                epsilon += wt[i]
                error_train_count = error_train_count + 1

        error_train = epsilon

        error_array[t][0] = error_train

        if epsilon > 0:
            alpha = 0.5*np.log((1-epsilon)/epsilon)

        c_wrong = np.exp(alpha)
        c_correct = np.exp(alpha*-1)
    

        for i in range(len(y_error)):
            if y_error[i] == -1:
                wt[i] = wt[i]*c_wrong
            else: 
                wt[i] = wt[i]*c_correct    
        wt = wt/sum(wt)

        dp_w_trace[t][0] = wt[0]
        dp_w_trace[t][1] = wt[27]
        dp_w_trace[t][2] = wt[75]

        alpha_epsilon[t][0] = epsilon
        alpha_epsilon[t][1] = alpha
        
        for i in range(len(X_test)):
            if float((X_test[i]*w.T)) >= 0:
              y_pred_test[i] += 1
            else:
              y_pred_test[i] += -1   

        y_error_test = y_pred_test * y_test 

        error_test_count = 0
        for i in range(len(y_error_test)):
            if y_error_test[i] < 0:
                error_test_count = error_test_count + 1

        error_test = error_test_count/float(len(y_error_test))

        error_array[t][1] = error_test


    plt.plot(error_array)
    plt.show()

    plt.plot(alpha_epsilon)
    plt.show()

    plt.plot(dp_w_trace)
    plt.show()
    #for i in range(500):
        #print 'point ',i, ' ', wt[i]

def binary_logit(X_train, y_train):
    eta = 0.1
    w = np.zeros([1,10])
    x = np.zeros([1,10])


    for i in range(500):
        x = X_train[i]*y_train[i]
        step = (1 - sigmoid(x,w.T) * x)
        w += eta/500 * step
    return w
        


    


def binary_logit_noboosting(X_train, X_test, y_train, y_test):

    w = np.matrix(binary_logit(X_train,y_train))
    y_pred = np.zeros([len(X_test),1])

    for i in range(len(X_test)):
        p = X_test[i]*w.T
        if p > 1:
            y_pred[i] = 1
        else:
            y_pred[i] = -1    
        print p
    conf_matrix = confusion_matrix(y_pred,y_test)

    print conf_matrix

    trace = sum(conf_matrix[i][i] for i in range(conf_matrix.shape[0]))

    print 'accurray nobootstrap:', trace/float(len(y_test))
    

def sigmoid(x,w):
    return 1 / (1 + (np.exp (-1*(x.dot(w)))))    

def binary_logit_boosting(X_train, X_test, y_train, y_test):
    X_train = np.matrix(X_train)
    X_test  = np.matrix(X_test)
    y_train = np.array(y_train)
    y_test  = np.array(y_test)

    wt = np.empty(len(X_train)).astype('float')
    i = 1/float(len(X_train))
    wt.fill(i)    
    n = len(X_train)
    j = len(y_test)
    y_pred_test  =   np.zeros([j,1], dtype=int)
    y_pred_train =  np.zeros([n,1], dtype=int)
    error_array =   np.zeros([1000,2])
    alpha_epsilon = np.zeros([1000,2])
    dp_w_trace       = np.zeros([1000,3])


    for t in range(1000):
        print "this is loop number: ", t
        bootstrap_X = np.zeros([n,10])
        bootstrap_y = np.zeros([n,1], dtype=int)
        u = uniform(0, max(wt))

        for i in range(n):
            sample = wt>=u
            pick = random.sample(np.arange(np.sum(sample)), 1)        

            bootstrap_X[i] = X_train[sample][pick]
            bootstrap_y[i] = y_train[sample][pick]
            u = uniform(0, wt[sample][pick])

        w = binary_logit(bootstrap_X, bootstrap_y)  

       

        for i in range(n):
            p = X_train[i]*w.T
            if p >= 1:
                y_pred_train[i] += 1
            else:
                y_pred_train[i] += -1   

        y_error = y_pred_train * y_train
    
        epsilon = 0
        error_train_count = 0
        for i in range(len(y_error)):
            if y_error[i] < 0:
                epsilon += wt[i]
                error_train_count = error_train_count + 1

        error_train = epsilon

        error_array[t][0] = error_train

        if epsilon > 0:
            alpha = 0.5*np.log((1-epsilon)/epsilon)

        c_wrong = np.exp(alpha)
        c_correct = np.exp(alpha*-1)
    

        for i in range(len(y_error)):
            if y_error[i] == -1:
                wt[i] = wt[i]*c_wrong
            else: 
                wt[i] = wt[i]*c_correct    
        wt = wt/sum(wt)

        dp_w_trace[t][0] = wt[0]
        dp_w_trace[t][1] = wt[27]
        dp_w_trace[t][2] = wt[75]

        alpha_epsilon[t][0] = epsilon
        alpha_epsilon[t][1] = alpha
        
        for i in range(len(X_test)):
            p = X_test[i]*w.T
            if p >= 1:
              y_pred_test[i] += 1
            else:
              y_pred_test[i] += -1   

        y_error_test = y_pred_test * y_test 

        error_test_count = 0
        for i in range(len(y_error_test)):
            if y_error_test[i] < 0:
                error_test_count = error_test_count + 1

        error_test = error_test_count/float(len(y_error_test))

        error_array[t][1] = error_test


    plt.plot(error_array)
    plt.show()

    plt.plot(alpha_epsilon)
    plt.show()

    plt.plot(dp_w_trace)
    plt.show()

def main():
    X = pd.read_csv('/home/lfawaz/Downloads/cancer_csv/X.csv', header=None)
    y = pd.read_csv('/home/lfawaz/Downloads/cancer_csv/y.csv', header=None)

    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=182,train_size=500,random_state=45)


    bayes_boosting(X_train, X_test, y_train, y_test)

    X_train = np.zeros([500,10])
    y_train = np.zeros([500,1])
    X_test  = np.zeros([183,10])
    y_test  = np.zeros([183,1])

    c = 0
    for i in range(len(X)):
        if i <= 182:
         X_test[i] = X[i]
         y_test[i] = y[i]
        if i >= 183:
         X_train[i-183] = X[i]
         y_train[i-183] = y[i]
  
    return_bayes_noboosting(X_train, X_test, y_train, y_test)


    binary_logit(X_train,y_train)

    binary_logit_noboosting(X_train, X_test, y_train, y_test)

    binary_logit_boosting(X_train, X_test, y_train, y_test)

main()
