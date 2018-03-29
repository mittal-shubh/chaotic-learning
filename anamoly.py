# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 05:33:49 2018

@author: ShubhM
"""

#from _future_ import division
from itertools import count
import matplotlib.pyplot as plt
from numpy import loadtxt
import numpy as np
import pandas as pd
import collections
from matplotlib import style
style.use('fivethirtyeight')
#%matplotlib inline

data = loadtxt("C:/Users/ShubhM/dataset/sunspots.txt", float)

df = pd.DataFrame(data, columns=['Months', 'Sunspots'])
#index = np.arange(0,len(df[df.columns[1]]))
#print(df.head())

def moving_average(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    #print(window)
    return np.convolve(data, window, 'same')

'''mov_avg = pd.DataFrame(moving_average(df[df.columns[1]], 10), 
                       columns=['Mov_Avg_SunSpots'])
residual = pd.DataFrame(df[df.columns[1]]-mov_avg[mov_avg.columns[0]], 
                        columns=['Residual'])
plt.plot(df[df.columns[1]], color='red')
plt.plot(mov_avg, color='blue')
plt.plot(residual, color='green')
plt.legend(loc='best')
plt.show()'''
   
def explain_anomalies(y, window_size, sigma=1.0, applying_rolling_std=False):
    avg = moving_average(y, window_size)
    avg_list = avg.tolist()
    residual = y-avg
    testing_std = residual.rolling(window=window_size).std()
    testing_std_as_df = pd.DataFrame(testing_std)
    rolling_std = testing_std_as_df.replace(np.nan, 
                testing_std_as_df.iloc[window_size-1,0]).round(3).iloc[:,0].tolist()
    std = np.std(residual)
    if applying_rolling_std:
        return {'standard_deviation': round(std, 3),
                'anomalies_dict': collections.OrderedDict([(index, y_i)for 
                                                           index, y_i, 
                                                           avg_i, rs_i 
                                                           in zip(count(), 
                                                                  y, avg_list, 
                                                                  rolling_std)
                                        if (y_i > avg_i + (sigma * rs_i)) | 
                                        (y_i < avg_i - (sigma * rs_i))])}
    else:
        return {'standard_deviation': round(std, 3),
                'anomalies_dict': collections.OrderedDict([(index, y_i) for
                                                       index, y_i, avg_i in 
                                                       zip(count(), y, avg) 
                                        if (y_i > avg_i + (sigma*std)) | 
                                        (y_i < avg_i - (sigma*std))])}

def plot_results(x, y, window_size, sigma_value=1,
                 text_xlabel="X Axis", text_ylabel="Y Axis", 
                 applying_rolling_std=False):
    """ Helps in generating the plot and flagging the anamolies.
        Supports both moving and stationary standard deviation. Use the 'applying_rolling_std' to switch
        between the two.
    Args:
    -----
        x (pandas.Series): dependent variable
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma_value (int): value for standard deviation
        text_xlabel (str): label for annotating the X Axis
        text_ylabel (str): label for annotatin the Y Axis
        applying_rolling_std (boolean): True/False for using rolling vs stationary standard deviation
    """
    plt.figure(figsize=(15, 8))
    plt.plot(x, y, "k.")
    y_av = moving_average(y, window_size)
    plt.plot(x, y_av, color='green')
    plt.xlim(-50, len(y))
    plt.xlabel(text_xlabel)
    plt.ylabel(text_ylabel)

    # Query for the anomalies and plot the same
    events = {}
    events = explain_anomalies(y, window_size, sigma=sigma_value, 
                               applying_rolling_std=applying_rolling_std)
    x_anomaly = np.fromiter(events['anomalies_dict'].keys(), dtype=int, 
                            count=len(events['anomalies_dict']))
    y_anomaly = np.fromiter(events['anomalies_dict'].values(), dtype=float,
                            count=len(events['anomalies_dict']))
    plt.plot(x_anomaly, y_anomaly, "r*", markersize=9)

    # add grid and lines and enable the plot
    #plt.grid(True)
    plt.show()
    
# 4. Lets play with the functions
x = df['Months']
y = df['Sunspots']

# plot the results
plot_results(x, y, window_size=10, text_xlabel="Months", sigma_value=3,
             text_ylabel="No. of Sun spots")
events = explain_anomalies(y, window_size=10, sigma=3)
# Display the anomaly dict
#print("Information about the anomalies model:{}".format(events))

# plot the results
plot_results(x, y, window_size=10, text_xlabel="Months", sigma_value=3,
             text_ylabel="No. of Sun spots", applying_rolling_std=True)
events_roll = explain_anomalies(y, window_size=10, sigma=3)
# Display the anomaly dict
#print("Information about the anomalies model:{}".format(events_roll))