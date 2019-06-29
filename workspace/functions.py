import numpy as np
import pylab as pl
from numpy.linalg import inv
import time

def heatmap_random(data,n):
    #
    
    w0 = np.zeros(n)
    w1 = np.zeros(n)
    
    for a in range(0,n):
        data_raveled = data.ravel()
        data_sum = np.sum(data_raveled)
        q = 0
        o = np.random.uniform(0,1) * data_sum
        e = 0
        i=1
        while e==0:
            q += data_raveled[i]
            if (q > o):
                e = i
                break
            i+=1
        w1[a] = -1 + (e/100 - (e%100)/100)*0.02
        w0[a] = -1 + (e%100)*0.02
    return w0, w1

def basis(x,mu,s):
    #the function for evaluating a basis function given a point in N-dimensional input space and also a mean and a variance.
    D = np.shape(mu)[0]
    magnitude = 0
    for a in range(0,D):
        magnitude += (x[a] - mu[a])**2
    magnitude = np.sqrt(magnitude)
    phi = np.exp(-(magnitude**2)/(2*(s**2)))
    return phi

def com_gauss_proc_kern(x_n, x_m, theta0, theta1, theta2, theta3):
    #the name is a shortened form of "common Gaussian process kernel".
    #This function calculates equation 6.63 for given parameters and 2 given x vectors.
    #Note: x_n and x_m has to be arrays. It has to be defined with functions like "np.array" or
    #"np.atleast_2d". Declarations like "x = 1" or "x = [3,3]" will not work.
    #The theta arguments are single values
    x_n = x_n.ravel()
    x_m = x_m.ravel()
    term1 = theta0*np.exp( (-theta1/2)*((np.sqrt( ((np.subtract(x_m,x_n))**2).sum()  ))**2)  )
    result = term1 + theta2 + theta3*np.sum(x_n*x_m)
    return result

def gauss_proc_kern_reg_cov(x_train, x_test, theta0, theta1, theta2, theta3, beta):
    #the name is a shortened form of "Gaussian process kernel regression covariance"
    #x_train is a an (n, d) array, where d is the number of dimensions and n is the number of training points.
    #if x_train is one-dimensional, use np.atleast_2d on it before passing it ti this definition
    #returns: cov_N, k, k_T, c, cov_full, inv_cov_N
    n, d = np.shape(x_train)
    x_full = np.append(x_train,np.atleast_2d(x_test)).reshape(n+1,d)
    cov = np.zeros([n+1,n+1])
    for a in range(0,n+1):
        for b in range(0,n+1):
            cov[a,b] = com_gauss_proc_kern(x_full[a,:], x_full[b,:], theta0, theta1, theta2, theta3 )
            if (a==b):
                cov[a,b] += (1/beta)
    inv_cov_N = inv(cov[:n,:n])
    return cov[:n,:n], cov[:n,n], cov[n,:n], cov[n,n], cov, inv_cov_N

def gauss_proc_kern_reg_pred_mean(k_T,C_N,t, inv_cov_N):
    #the name is a shortened form of "Gaussian process kernel regression prediction mean"
    o = np.dot(k_T,inv_cov_N)
    i = np.dot(np.atleast_2d(o),np.atleast_2d(t.ravel()).transpose() )
    return i

def gauss_proc_kern_reg_pred_var(c,k_T,C_N,k, inv_cov_N):
    #the name is a shortened form of "Gaussian process kernel regression prediction variance"
    o = np.dot(k_T,inv_cov_N)
    i = c - np.dot(np.atleast_2d(o),k )
    return i 

def theta0_deriv(x_n, x_m, theta0, theta1):
    x_n = x_n.ravel()
    x_m = x_m.ravel()
    result = np.exp( (-theta1/2)*((np.sqrt( ((np.subtract(x_m,x_n))**2).sum()  ))**2)  )
    return result
def theta1_deriv(x_n, x_m, theta0, theta1):
    x_n = x_n.ravel()
    x_m = x_m.ravel()
    result1 = theta0*np.exp( (-theta1/2)*((np.sqrt( ((np.subtract(x_m,x_n))**2).sum()  ))**2)  )
    result = result1*(-1/2)*((np.sqrt( ((np.subtract(x_m,x_n))**2).sum()  ))**2)
    return result
def theta3_deriv(x_n, x_m, theta3):
    x_n = x_n.ravel()
    x_m = x_m.ravel()
    result = np.sum(x_n*x_m)
    return result

def gauss_proc_kern_reg_deriv_values(x_train, theta0, theta1, theta2, theta3, beta):
    #the name is a shortened form of "Gaussian process kernel regression derivative values"
    #It returns all the values needed to solve equation (6.70)
    #x_train is a an (n, d) array, where d is the number of dimensions and n is the number of training points.
    #if x_train is one-dimensional, use np.atleast_2d on it before passing it ti this definition
    #returns: cov_N, k, k_T, c, cov_full
    n, d = np.shape(x_train)
    cov = np.zeros([n,n])
    cov_theta0_deriv = np.zeros([n,n])
    cov_theta1_deriv = np.zeros([n,n])
    cov_theta2_deriv = np.zeros([n,n])
    cov_theta3_deriv = np.zeros([n,n])
    for a in range(0,n):
        for b in range(0,n):
            cov[a,b] = com_gauss_proc_kern(x_train[a,:], x_train[b,:], theta0, theta1, theta2, theta3 )
            cov_theta0_deriv[a,b] = theta0_deriv(x_train[a,:], x_train[b,:], theta0, theta1 )
            cov_theta1_deriv[a,b] = theta1_deriv(x_train[a,:], x_train[b,:], theta0, theta1 )
            cov_theta2_deriv[a,b] = 1
            cov_theta3_deriv[a,b] = theta3_deriv(x_train[a,:], x_train[b,:], theta3 )
            if (a==b):
                cov[a,b] += (1/beta)
    inv_cov = inv(cov) #inverse of the matrix covariance
    return cov, inv_cov, cov_theta0_deriv, cov_theta1_deriv, cov_theta2_deriv, cov_theta3_deriv

def gauss_proc_kern_reg_deriv(x_train, t_train, inv_cov, cov_theta_deriv):
    #Implementation of equation 6.70 in [Bishop, 2006] 
    term1 = (-1/2)*np.trace(np.dot(inv_cov,cov_theta_deriv))
    term2 = np.dot(t_train.transpose(),inv_cov)
    term2 = np.dot(term2,cov_theta_deriv)
    term2 = np.dot(term2,inv_cov)
    term2 = (-1/2)*np.dot(term2,t_train)
    result = term1 + term2
    return result
def optim_kern_param(theta0,theta1,theta2,theta3,beta,N,x_train,t_train):
    #optmize kernel parameters
    #for the specified kernel (Bishop equation 6.63)
    
    #arrays to hold the values of the parameters after each iteration
    theta0_array = [theta0]
    theta1_array = [theta1]
    theta2_array = [theta2]
    theta3_array = [theta3]

    #variables to hold the value of the parameters after each iteration. They are initially given the prior values.
    theta0_new = theta0
    theta1_new = theta1
    theta2_new = theta2
    theta3_new = theta3
    
    for a in range(0,N):
        q,w,e,r,t,y = gauss_proc_kern_reg_deriv_values(x_train, theta0_new, theta1_new, theta2_new, theta3_new, beta)
        cov = q
        inv_cov = w
        cov_theta0_deriv = e
        cov_theta1_deriv = r
        cov_theta2_deriv = t
        cov_theta3_deriv = y
        theta0_grad = gauss_proc_kern_reg_deriv(x_train, t_train, inv_cov, cov_theta0_deriv)
        theta1_grad = gauss_proc_kern_reg_deriv(x_train, t_train, inv_cov, cov_theta1_deriv)
        theta2_grad = gauss_proc_kern_reg_deriv(x_train, t_train, inv_cov, cov_theta2_deriv)
        theta3_grad = gauss_proc_kern_reg_deriv(x_train, t_train, inv_cov, cov_theta3_deriv)
        theta0_new -= theta0_grad
        theta1_new -= theta1_grad
        theta2_new -= theta2_grad
        theta3_new -= theta3_grad
        theta0_array = np.append(theta0_array,theta0_new)
        theta1_array = np.append(theta1_array,theta1_new)
        theta2_array = np.append(theta2_array,theta2_new)
        theta3_array = np.append(theta3_array,theta3_new)
    
    return theta0_new,theta1_new,theta2_new,theta3_new,theta0_array,theta1_array,theta2_array,theta3_array

def lin_reg_pred_distrib(train_x,train_y,test_x,test_y,M,alpha,beta,s):
    #Linear regression predictive distribution
    #inputs:
    # train_x: the input of the training data
    # train_y: the output of the training data
    # test_x: the input of the test data
    # test_x: the output of the test data
    # M: basis function ticks for each dimension (i.e. M^D basis functions)
    # alpha: hyperparameter
    # beta: hyperparameter
    # s: the variance of the Gaussian basis functions
    #outputs:
    # pred_means: an array of the mean of the prediction for each test point
    # pred_means: the average of pred_means
    # test_pred_avg_error: the average error of the prediction
    # test_pred_error_vari: the variance of test_pred_avg_error
    # calc_time: the time the this definition took to run.
    
    D = np.shape(train_x)[1]
    
    c1 = time.process_time() 
    
    means = np.zeros((M**D,D))
    done = 0
    mu_vector_amount = M**D
    row_count = 0
    column_count = 0
    for a in range(0,mu_vector_amount*D):
        means[row_count,column_count] = ( (row_count//(M**column_count))%M*2/(M-1)) - 1
        row_count += 1
        if (row_count == mu_vector_amount):
            row_count = 0
            column_count += 1
    
    design_matrix = np.zeros((np.shape(train_x)[0],M**D))
    for a in range(0,M**D):
        for b in range(0,np.shape(train_x)[0]):
            design_matrix[b,a] = basis(train_x[b,:],means[a,:],s)
    S_N = inv( alpha*(np.identity(M**D)) + beta*np.dot(design_matrix.transpose(),design_matrix) )
    m_N = beta * (   np.dot( np.dot(S_N,design_matrix.transpose()), np.atleast_2d(train_y).transpose() ) )
    
    pred_means = np.zeros(np.shape(test_x)[0])
    for b in range(0,np.shape(test_x)[0]):

        phi_vector_pred = np.zeros(M**D)
        for a in range(0,M**D):
            phi_vector_pred[a] = basis(test_x[b,:],means[a],s)
        phi_vector_pred = np.atleast_2d(phi_vector_pred).transpose()

        pred_means[b] = np.dot( m_N.transpose(), phi_vector_pred )
        
    c2 = time.process_time()
    calc_time = c2 - c1
    
    pred_means_avg = np.average(pred_means)
    
    #test_y = test_y.reshape(100,100).transpose()
    #test_y = test_y.ravel()
    
    test_pred_error = np.abs(pred_means-test_y)
    test_pred_avg_error = np.average(test_pred_error)
    
    test_pred_error_dev = np.abs(test_pred_error - test_pred_avg_error)
    test_pred_error_vari = np.average(test_pred_error_dev)
    
    return pred_means, pred_means_avg, test_pred_avg_error, test_pred_error_vari, calc_time


def Gauss_proc_kern_reg(train_x,train_y,test_x,test_y,N,beta,theta0,theta1,theta2,theta3):
    #Gaussian process kernel regression
    #inputs:
    # train_x: the input of the training data
    # train_y: the output of the training data
    # test_x: the input of the test data
    # test_x: the output of the test data
    # N: number of iterations of optimisation
    # beta: parameter
    # theta0: parameter
    # theta1: parameter
    # theta2: parameter
    # theta3: parameter
    #outputs:
    # pred_means: an array of the mean of the prediction for each test point
    # pred_means: the average of pred_means
    # test_pred_avg_error: the average error of the prediction
    # test_pred_error_vari: the variance of test_pred_avg_error
    # calc_time: the time the this definition took to run.
    
    c1 = time.process_time() 
    
    N_train = np.shape(test_y)[0]
    
    e1,e2,e3,e4,e5,e6,e7,e8 = optim_kern_param(theta0,theta1,theta2,theta3,beta,N,train_x,train_y)
    theta0 = e1; theta1 = e2; theta2 = e3; theta3 = e4
    theta0_array = e5; theta1_array = e6; theta2_array = e7; theta3_array = e8
    
    pred_means = np.zeros(N_train)
    pred_vars = np.zeros(N_train)
    for a in range(0,N_train):
        C_N, k, k_T, c, C_N_full, inv_cov_N = gauss_proc_kern_reg_cov(train_x,test_x[a,:],theta0,theta1,theta2,theta3,beta)
        pred_means[a] = gauss_proc_kern_reg_pred_mean(k_T,C_N,train_y, inv_cov_N)
        pred_vars[a] = gauss_proc_kern_reg_pred_var(c,k_T,C_N,k, inv_cov_N)
    
    c2 = time.process_time()
    calc_time = c2 - c1
    
    pred_means_avg = np.average(pred_means)
    
    test_pred_error = np.abs(pred_means-test_y)
    test_pred_avg_error = np.average(test_pred_error)
    
    test_pred_error_dev = np.abs(test_pred_error - test_pred_avg_error)
    test_pred_error_vari = np.average(test_pred_error_dev)
    
    return pred_means, pred_means_avg, test_pred_avg_error, test_pred_error_vari, calc_time





