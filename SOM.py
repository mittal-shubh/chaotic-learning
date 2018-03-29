# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 02:11:14 2018

@author: ShubhM
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#from matplotlib import patches as patches

np.random.seed(1000)

#Reading the data
raw_data = pd.read_csv("C:/Users/ShubhM/Desktop/EE671A/Project/data.csv")

#Normalize dataset
def scale(x):
    centred = x/x.max()
    return centred
#print(raw_data[raw_data.columns[2]])
raw_data[[raw_data.columns[2],raw_data.columns[3],raw_data.columns[4]]]=raw_data[
        [raw_data.columns[2],raw_data.columns[3],
         raw_data.columns[4]]].apply(scale)
raw_data[[raw_data.columns[0],raw_data.columns[1],raw_data.columns[5], 
          raw_data.columns[6]]]=raw_data[[raw_data.columns[0],raw_data.columns[1],
                          raw_data.columns[5], raw_data.columns[6]]].apply(scale)

#Shuffling the data
#raw_data = raw_data.sample(frac=1).reset_index(drop=True)

inp_data = raw_data.drop(raw_data.columns[[0,1]], axis=1)
out_data = raw_data.drop(raw_data.columns[[2,3,4,5,6]], axis=1)
raw_data = raw_data.T
inp_data = inp_data.T
out_data = out_data.T

#Network Parameters initialization
network_dimensions = np.array([5, 5])
n_iterations = 7000
init_learning_rate_w = 0.9
fin_learning_rate_w = 0.1
init_learning_rate_t = 0.9
fin_learning_rate_t = 0.9
init_learning_rate_a = 0.9
fin_learning_rate_a = 0.9
#print(network_dimensions)

# establish size variables based on data
m = inp_data.shape[0]
n = raw_data.shape[1]
o = out_data.shape[0]

# weight matrix (i.e. the SOM) needs to be one m-dimensional vector for each neuron in the SOM
net_w = np.random.random((network_dimensions[0], network_dimensions[1], m))
net_t = np.random.random((network_dimensions[0], network_dimensions[1], o))
net_a = np.random.random((network_dimensions[0], network_dimensions[1], o, m))
#print(net_w[0][0], net_o[0][0], net_A[0][0])

# initial neighbourhood radius
#init_radius = max(network_dimensions[0], network_dimensions[1])/2
init_radius = 2.5
fin_radius = 1
#radius decay parameter
#time_constant = n_iterations / np.log(init_radius)

def find_bmu(t, net_w, m):
    """
        Find the best matching unit for a given vector, t, in the SOM
        Returns: a (bmu, bmu_idx) tuple where bmu is the high-dimensional BMU
                 and bmu_idx is the index of this vector in the SOM
    """
    bmu_idx = np.array([0, 0])
    # set the initial minimum distance to a huge number
    min_dist = np.iinfo(np.int).max
    # calculate the high-dimensional distance between each neuron and the input
    for x in range(net_w.shape[0]):
        for y in range(net_w.shape[1]):
            w = net_w[x, y, :].reshape(m, 1)
            # don't bother with actual Euclidean distance, to avoid expensive sqrt operation
            sq_dist = np.sqrt(np.sum((w - t) ** 2))
            if sq_dist < min_dist:
                min_dist = sq_dist
                bmu_idx = np.array([x, y])
    # get vector corresponding to bmu_idx
    bmu = net_w[bmu_idx[0], bmu_idx[1], :].reshape(m, 1)
    # return the (bmu, bmu_idx) tuple
    return (bmu, bmu_idx)

#Fn for decaying the radius
def decay_radius(initial_radius, final_radius, i, n_iterations):
    return initial_radius * ((final_radius/initial_radius)**(i / n_iterations))

#Fn for decaying the learning rates
def decay_learning_rate(initial_learning_rate, final_learning_rate, i, n_iterations):
    return initial_learning_rate * ((final_learning_rate/initial_learning_rate)**(i / n_iterations))

#Gaussian Neighborhood Function
def calculate_influence(distance, radius):
    return np.exp(-(distance**2) / (2* (radius**2)))

#Converting dfs into numpy arrays
raw_data = np.array(raw_data)

def networkperformance(raw_data, net_w, net_a, net_t, r):
    tn_x = np.array(0)
    tn_y = np.array(0)
    n = raw_data.shape[1]
    for i in range(n):
        print(i)
        #inputs
        t_raw = raw_data[:, i].reshape(np.array([raw_data.shape[0], 1]))
        ti = t_raw[2:, :]
        
        #find the winner neuron w
        w, w_idx = find_bmu(ti, net_w, m)
        #print(w, w_idx)
    
        #Winning Neuron Output
        #tw = net_t[w_idx[0]][w_idx[1]] + np.dot(net_a[w_idx[0]][w_idx[1]],ti-w)
    
        #Network Output
        nt = np.zeros(2)
        nt = nt.reshape(o,1)
        s = 0
        for x in range(net_w.shape[0]):
            for y in range(net_w.shape[1]):
                w = net_w[x, y, :].reshape(m, 1)
                theta = net_t[x, y, :].reshape(o, 1)
                w_dist = np.sqrt(np.sum((np.array([x, y]) - w_idx) ** 2))
                #print(r)
                if w_dist <= r:
                    influence = calculate_influence(w_dist, r)
                    s = s + influence
                    nt = nt + influence*(theta+np.dot(net_a[x][y],(ti-w)))
        tn_x = np.append(tn_x, nt[0]/s)
        tn_y = np.append(tn_y, nt[1]/s)
        
    tn = np.append([tn_x[1:]], [tn_y[1:]], axis=0)
    #true outputs
    to = raw_data[:2, :]
    return tn, to  

#Test Data
test_data = pd.read_csv("C:/Users/ShubhM/Desktop/test3.csv")
#print(test_data)
#Normalize dataset
test_data[[test_data.columns[2],test_data.columns[3],test_data.columns[4]]]=test_data[
        [test_data.columns[2],test_data.columns[3],
         test_data.columns[4]]].apply(scale)
test_data[[test_data.columns[0],test_data.columns[1],test_data.columns[5], 
          test_data.columns[6]]]=test_data[[test_data.columns[0],
                           test_data.columns[1], test_data.columns[5], 
                           test_data.columns[6]]].apply(scale)
test_data = test_data.T
test_data = np.array(test_data)
#n = test_data.shape[1]

#Training of network
epochmse_x = np.array(0)
epochmse_y = np.array(0)
for i in range(n_iterations):
    print('Iter: %d' % i)
    
    # decay the SOM parameters
    r = decay_radius(init_radius, fin_radius, i, n_iterations)
    l_w = decay_learning_rate(init_learning_rate_w, fin_learning_rate_w, i, n_iterations)
    l_t = decay_learning_rate(init_learning_rate_t, fin_learning_rate_t, i, n_iterations)
    l_a = decay_learning_rate(init_learning_rate_a, fin_learning_rate_a, i, n_iterations)
    
    #select a training example at random (try stepwise after shuffling the data)
    t_raw = raw_data[:, np.random.randint(0, n)].reshape(np.array([raw_data.shape[0], 1]))
    '''if i >= 4998:
        j = i-4998
    else:
        j = i
    t_raw = raw_data[:, j].reshape(np.array([raw_data.shape[0], 1]))'''
    ti = t_raw[2:, :]
    to= t_raw[:2, :]
    #print(t_raw, ti, to)

    #find the winning neuron
    w, w_idx = find_bmu(ti, net_w, m)
    #print(w, w_idx)
    
    nt = np.zeros(2)
    nt = nt.reshape(o,1)
    s = 0
    for x in range(net_w.shape[0]):
        for y in range(net_w.shape[1]):
            w = net_w[x, y, :].reshape(m, 1)
            theta = net_t[x, y, :].reshape(o, 1)
            w_dist = np.sqrt(np.sum((np.array([x, y]) - w_idx) ** 2))
            if w_dist <= r:
                influence = calculate_influence(w_dist, r)
                s = s + influence
                nt = nt + influence*(theta+np.dot(net_a[x][y],(ti-w)))
    tn = nt/s   #network output = nt
    
    #now we know the BMU, update its weight vector to move closer to input
    #and move its neighbours in 2-D space closer
    #by a factor proportional to their 2-D distance from the BMU
    for x in range(net_w.shape[0]):
        for y in range(net_w.shape[1]):
            w = net_w[x, y, :].reshape(m, 1)
            theta = net_t[x, y, :].reshape(o, 1)
            a = net_a[x, y, :]     #******
            # get the 2-D distance (again, not the actual Euclidean distance)
            w_dist = np.sqrt(np.sum((np.array([x, y]) - w_idx) ** 2))
            # if the distance is within the current neighbourhood radius
            if w_dist <= r:
                # calculate the degree of influence (based on the 2-D distance)
                influence = calculate_influence(w_dist, r)
                # now update the neuron's weight using the formula:
                new_w = w + (l_w * influence * (ti - w))
                
                #update the theta
                delta_t = (influence/s)*(to-tn)
                new_t = theta + (l_t * delta_t)
                
                #update the j_matrix
                delta_a = (influence/s)*np.dot(to-tn, ti.T-w.T)
                new_a = a + (l_a * delta_a)
                
                # commit the new weight, theta and j_matrix
                net_w[x, y, :] = new_w.reshape(1, m)
                net_t[x, y, :] = new_t.reshape(1, o)
                net_a[x, y, :] = new_a
    #tn, to = networkperformance(raw_data, net_w, net_a, net_t, r)
    #epochmsei = ((to-tn) ** 2).mean(axis=1)
    #epochmse_x = np.append(epochmse_x, epochmsei[0]/s)
    #epochmse_y = np.append(epochmse_y, epochmsei[1]/s)


#Network Performance for training data
tn, to = networkperformance(raw_data, net_w, net_a, net_t, r)
train_mse = ((to-tn) ** 2).mean(axis=1)
dfx = pd.DataFrame(data=np.column_stack((tn[0],to[0])),columns=['pred_Vx','true_Vx'])
dfy = pd.DataFrame(data=np.column_stack((tn[1],to[1])),columns=['pred_Vy','true_Vy'])
#UNCOMMENT FOR TESTING
#test.csv should have 0th row as index values and first two columns to be 
#the output velocities, next three sonar readings, and next two 
#previous time-step velocities
test_tn, test_to = networkperformance(test_data, net_w, net_a, net_t, r)
test_mse = ((test_to-test_tn) ** 2).mean(axis=1)
dftx = pd.DataFrame(data=np.column_stack((test_tn[0],test_to[0])),columns=['pred_Vx','true_Vx'])
dfty = pd.DataFrame(data=np.column_stack((test_tn[1],test_to[1])),columns=['pred_Vy','true_Vy'])

'''
#Plotting of Epoch error
epochmse = np.append([epochmse_x[1:]], [epochmse_y[1:]], axis=0)
dfex = pd.DataFrame(data=np.column_stack((epochmse[0],epochmse[1])),columns=['Vx','Vy'])
#Plots of tru vs observed velocities
dfex[dfex.columns[0]].plot(title='Vx', 
      sharex = False, figsize = (10,5), color = 'red')
plt.legend(loc='best')
plt.show()
dfex[dfex.columns[1]].plot(title='Vy', 
      sharex = False, figsize = (10,5), color = 'red')
plt.legend(loc='best')
plt.show()
'''

#Plots of true vs observed velocities (Training)
dfx[dfx.columns[0]].plot(title='Vx', 
      sharex = False, figsize = (15,8), color = 'red')
plt.legend(loc='best')
dfx[dfx.columns[1]].plot(title='Train Vx', 
      sharex = False, figsize = (15,8), color = 'blue')
plt.legend(loc='best')
plt.show()
dfy[dfy.columns[0]].plot(title='Vy', 
      sharex = False, figsize = (15,8), color = 'red')
plt.legend(loc='best')
dfy[dfy.columns[1]].plot(title='Train Vy', 
      sharex = False, figsize = (15,8), color = 'blue')
plt.legend(loc='best')
plt.show()
#Scatterplots for predicted vs observed velocities
#plt.scatter(to[0], tn[0], c="b", alpha=0.5,label="Predicted vs True Vx")
#plt.scatter(to[1], tn[1], c="r", alpha=0.5,label="Predicted vs True Vy")
#plt.show()

#Plots of true vs observed velocities (Testing)
dftx[dftx.columns[0]].plot(title='Vx', 
      sharex = False, figsize = (15,8), color = 'red')
plt.legend(loc='best')
dftx[dftx.columns[1]].plot(title='Test Vx', 
      sharex = False, figsize = (15,8), color = 'blue')
plt.legend(loc='best')
plt.show()
dfty[dfty.columns[0]].plot(title='Vy', 
      sharex = False, figsize = (15,8), color = 'red')
plt.legend(loc='best')
dfty[dfty.columns[1]].plot(title='Test Vy', 
      sharex = False, figsize = (15,8), color = 'blue')
plt.legend(loc='best')
plt.show()
#Scatterplots for predicted vs observed velocities
#plt.scatter(test_to[0], test_tn[0], c="b", alpha=0.5,label="Predicted vs True Vx")
#plt.scatter(test_to[1], test_tn[1], c="r", alpha=0.5,label="Predicted vs True Vy")
#plt.show()


#Printing the train and test mean square error
print("Training Mean Square Error for Vx & Vy: %2f %2f" %(train_mse[0], train_mse[1]))
print("Test Mean Square Error for Vx & Vy: %2f %2f" %(test_mse[0], test_mse[1]))