# -*- coding: utf-8 -*-
"""
@author: Venkata praveen Kumar Madhavarapu
"""


import csv
import pandas as pd
import numpy as np
from scipy.stats import hmean
from scipy.special import boxcox, inv_boxcox
from scipy import stats
import random
from sklearn.cluster import KMeans
from pandas.plotting import scatter_matrix
from scipy.stats import norm
from matplotlib import pyplot as plt
import math
#import tensorflow as tf
import seaborn as sns
sns.set()




train_df = pd.read_csv("\\Electricity data\\TOSG_cont\\2015.csv", header = 0)           # load the dataset into train_df 
train_df.use = train_df.use * 1000                                                      # converting the meter readings from Kilo Watts to Watts


more =  train_df[train_df['use'] > 3000]
less =  train_df[train_df['use'] <= 3000]
more['use'] = 3000
train_df1 = more.append(less)                                                           # ignoring readings above 3000 Watts

       
train_df2 = train_df1.copy()


train_df2.use = stats.boxcox(train_df2.use, lmbda=0.35)                                 # boxcox transformation

final_df = train_df2.use.groupby(train_df.localminute).unique().apply(pd.Series)        # grouping by time for time series data 



hm_avg = []                 # daily harmonic mean
am_avg = []                 # daily arithmetic mean
sd_avg = []                 # daily standard deviation
Q_ratio = []                # Anomaly detection metric  

for i in range(330):
    hm = []                 # hourly harmonic mean
    am = []                 # hourly arithmetic mean
    sd = []                 # hourly standard deviation
    for j in range(24):
        hm.append(hmean(final_df.iloc[i*24+j,:].dropna()))
        am.append(np.mean(final_df.iloc[i*24+j,:].dropna()))
        sd.append(np.std(final_df.iloc[i*24+j,:].dropna()))
    hm_avg.append(np.mean(hm))
    am_avg.append(np.mean(am))
    sd_avg.append(np.mean(sd))
    Q_ratio.append(hm_avg[i]/am_avg[i])                                         # hm/am daily
    


##########################   injecting false data    ##########################

Smart_Meters = np.unique(train_df['dataid'])                                    # set of unique meter ids
N = np.size(Smart_Meters)                                                       # Number of smart meters
Rho_mal = 30                                                                    # Percentage of compromised meters
M = np.ceil(Rho_mal/100*N)                                                      # Number of compromised meters (Rho_mal = M/N)
Frame = 30                                                                      # Size of frame in days
window = 30*24                                                                  #N umber of time slots in frame


deltaAvg = 500
deltaMin = 450
deltaMax = 550

false_df = train_df1.copy()

rand_50 = pd.DataFrame(np.random.randint(0,50,size=(len(false_df), 1)), columns=['rand'])
 
false_df['use'] = false_df['use'] + deltaAvg + rand_50['rand']




false_df.use = stats.boxcox(false_df.use, lmbda=0.5)

false_final_df = false_df.use.groupby(train_df.localminute).unique().apply(pd.Series)



for i in range(60):
    hm1=[]
    am1=[]
    sd1=[]
    for j in range(24):
        hm1.append(hmean(false_final_df.iloc[i*24+j,:].dropna()))
        am1.append(np.mean(false_final_df.iloc[i*24+j,:].dropna()))
        sd1.append(stats.median_absolute_deviation(false_final_df.iloc[i*24+j,:].dropna()))
    hm_avg.append(np.mean(hm1))
    am_avg.append(np.mean(am1))
    sd_avg.append(np.mean(sd1))
    Q_ratio.append(hm_avg[i]/am_avg[i]) 
    

mean_correction=[]
for i in range(30):
    mean_correction.append(hm_avg[i] - (am_avg[i] - hm_avg[i]))


meterwise_original = train_df2.use.groupby(train_df.dataid).unique().apply(pd.Series)

meterwise_falsified = false_df.use.groupby(train_df.dataid).unique().apply(pd.Series)



frames = [meterwise_original[:214][:], meterwise_falsified[214:][:]]
#frames = [meterwise_original[:150][:], meterwise_falsified[150:][:]]
meterwise = pd.concat(frames)


##########################   Gaussian Trust Model    ##########################

temp = np.zeros(shape=(N,Window))
theta = pd.DataFrame(temp)
l_original = pd.DataFrame(temp)
x = l_original.copy()
cw = l_original.copy()
w = l_original.copy()

for i in range(N):
    for j in range(Window):
        theta.iloc[i,j] = abs(meterwise.iloc[i,j] - am_avg[math.floor(j/24)])
        if theta.iloc[i,j]<sd_avg[math.floor(j/24)]:
            l_original.iloc[i,j]=4
        elif theta.iloc[i,j]<2*sd_avg[math.floor(j/24)]:
            l_original.iloc[i,j]=3
        elif theta.iloc[i,j]<3*sd_avg[math.floor(j/24)]:
            l_original.iloc[i,j]=2
        else:
            l_original.iloc[i,j]=1
    

l1_original = l_original.apply(np.sort, axis = 1)
 
K=4
 
for i in range(N):
    for j in range(Window):
        x.iloc[i,j] = 1 + ((K-1)*j)/Window
        
temp1 = np.zeros(shape=N)
std_dr = pd.DataFrame(temp1)
   
for i in range(N):
    std_dr.iloc[i] = np.std(l1_original[i])    
    
for i in range(len(std_dr)):
    if int(std_dr.iloc[i]) == 0:
       std_dr.iloc[i] = np.mean(std_dr) 

M_BR = 4

for i in range(N):
    for j in range(Window):
        cw.iloc[i,j] = (1/(math.sqrt(2*3.1415)*std_dr.iloc[i,0]))*(math.exp((-1*math.pow((x.iloc[i,j]-M_BR),2))/(2*math.pow(std_dr.iloc[i,0],2))))
        

for i in range(N):
    for j in range(Window):
        w.iloc[i,j] = cw.iloc[i,j]/np.sum(cw.iloc[i,:])
        
eeta = 2;
R = np.zeros(shape=N)


for meter in range(N):
    temp2 = np.zeros(shape=(4,Window))
    I = pd.DataFrame(temp2)
    

#creates the evasion discrete levels.
    for j in range(Window):
        if l_original.iloc[meter,j] == 1:
            I.iloc[0,j] = 1
        elif l_original.iloc[meter,j] == 2:
            I.iloc[1,j] = 1
        elif l_original.iloc[meter,j] == 3:
            I.iloc[2,j] = 1
        else:
            I.iloc[3,j] = 1
            
    temp3 = np.zeros(shape=4)
    wd = pd.DataFrame(temp3)
    
    for i in range(4):
        for j in range(Window):
            wd.iloc[i,0] = wd.iloc[i,0] + I.iloc[i,j]*w.iloc[meter,j]
            
    for i in range(4):
        R[meter] = R[meter] + (i+1)*wd.iloc[i,0]

            


TR = np.zeros(shape=N)

for meter in range(N):
    TR[meter] = (1/math.pow(K,eeta))*(math.pow(R[meter],eeta))
    

##########################   Plotting the Scores    ##########################
xax = np.zeros(shape=N)

for i in range(N):
    xax[i] = i
    

plt.hold(True)
for i in range(N):
    if i<=M:
        if i==M:
            plt.plot(xax[i],TR[i],'bo', label = "Honest Smart Meter")           #for legend
        else:
            plt.plot(xax[i],TR[i],'bo')
            
    else:
        if i==N:
            plt.plot(xax[i],TR[i],'r*', label = "Malicious Smart Meter")        #for legend
        else:
            plt.plot(xax[i],TR[i],'r*')
        


plt.xlabel('Smart meter ID')
plt.ylabel('Trust Score')
plt.legend()
#plt.savefig('Attack_Evasion')


##########################   classification using kmeans    ##########################
kmeans = KMeans(n_clusters=2)
kmeans.fit(TR.reshape(-1,1))
y_kmeans = kmeans.predict(TR.reshape(-1,1))
md = np.sum(y_kmeans[150:]==1)
fa = np.sum(y_kmeans[:150]==0)
print(md/69)
print(fa/150)






