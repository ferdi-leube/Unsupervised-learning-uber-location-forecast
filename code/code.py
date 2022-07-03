# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:03:16 2022

@author: leube
"""

# import data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\leube\Downloads\uber-raw-data-jun14.csv")

# check data structure and layout

data.shape
data.info()
data.columns
data = data.sample(n=7000, random_state=42)
# check number of unique values and the unique values themselves for each column in data

for x in list(data.columns):
    print(x)
    print(data[x].nunique())
    print('\n')
    
    
    
for x in list(data.columns):
    print(x)
    print(data[x].unique())
    print('\n')
    
# statistical overview of numerical columns 
   
data.describe()

# check for missing values

data.isnull().sum()

# encode Base
from sklearn.preprocessing import LabelEncoder
tst = data.copy()

le = LabelEncoder()
data['Base'] = le.fit_transform(data.Base.values)

data['Base'].value_counts()

le = LabelEncoder()
data['Day'] = le.fit_transform(data.Base.values)

# for tst
Xtst = tst.drop('Date/Time',axis=1)
ytst = tst['Date/Time']

# devide data in x and y
data.columns
X = data[['Lat','Lon']]

# Hour and weekday column creation

data['Date/Time'] = data['Date/Time'].apply(pd.to_datetime)

data['Day'] = data['Date/Time'].dt.dayofweek
data['Hour'] = data['Date/Time'].dt.hour
# data['Hour_test'] = pd.to_datetime(data['Date/Time']).dt.hour


# EDA

sns.displot(data=data,x='Day')
sns.displot(data=data,x='Hour')




# KMEANS, use elbow method to look for optimal number of clusters
import sklearn
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import davies_bouldin_score

model = KMeans()
visualizer = KElbowVisualizer(estimator = model, k = (2,10))
visualizer.fit(X)
visualizer.poof()

km = KMeans(n_clusters=5)
result = km.fit(X)
predictk5 = result.predict(X)


model.cluster_centers_

data['KMEANS5_labels'] = model.labels_

# DBSCAN

from sklearn.cluster import DBSCAN
sdtst = data.copy()

#rescale?

modeldb = DBSCAN(eps=0.01, min_samples=20)
resultdb = modeldb.fit(X)
predictdb = result.predict(X)

set(modeldb.labels_)

from sklearn import metrics

# a form of hyper parameter tuning for dbcan, testing different inputs for eps and min_samples
# the user should here have the use case and the data in mind when deciding which inputs
# to test, lets create some functions!!


from sklearn.metrics import silhouette_score

def check_DBSCAN_params(X, eps=None, min_samples=None, epsc=None, minc=None):
    # eps and min_samples list with  variables you want to try
    
    if eps == None:
        eps=print(input('Please enter at least one value for eps: '))
    if min_samples == None: 
        min_samples = print(input('Please enter at least one value for min_samples: '))
    if epsc == True:
        scoressil={}
        scoresdav={}
        scores = []
        for ep in eps:
            model =  DBSCAN(eps=ep, min_samples = min_samples)
            model.fit(X)
            #predictions = result.predict(X)
            counter=collections.Counter(model.labels_)
            print(f'for this eps = {ep} frequencies per label = {counter}')
            if len(set(model.labels_))>1:
                scores.append(silhouette_score(X, model.labels_, metric="sqeuclidean"))
                scoressil[ep] = silhouette_score(X, model.labels_, metric="sqeuclidean")
                scoresdav[ep] = davies_bouldin_score(X, model.labels_)
    
        sns.lineplot(x=eps, y=scores)
        plt.title('eps')
        plt.show()
        return scoressil, scoresdav
    elif minc == True:
        scoressil={}
        scoresdav={}
        scores = []
        counter=collections.Counter(model.labels_)
        print(f'for this eps = {ep} frequencies per label = {counter}')
        for mini in min_samples:
            model =  DBSCAN(eps=eps, min_samples = mini)
            model.fit(X)
            #predictions = result.predict(X)
            if len(set(model.labels_))>1:
                scores.append(silhouette_score(X, model.labels_, metric="sqeuclidean"))
                scoressil[ep] = silhouette_score(X, model.labels_, metric="sqeuclidean")
                scoresdav[ep] = davies_bouldin_score(X, model.labels_)
    
        sns.lineplot(x=min_samples, y=scores)
        plt.title('min_samples')
        plt.show()
        return scoressil, scoresdav
    elif epsc == True and minc == True:
        scoressil={}
        scoresdav={}
        scores = []
        for mini in min_samples:
            for ep in eps:
                model =  DBSCAN(eps=ep, min_samples=mini)
                model.fit(X)
                #predictions = result.predict(X)
                counter=collections.Counter(model.labels_)
                print(f'for this eps = {ep} and this {mini} frequencies per label = {counter}')
                if len(set(model.labels_))>1:
                    scores.append(silhouette_score(X, model.labels_, metric="sqeuclidean"))
                    scoressil[ep] = silhouette_score(X, model.labels_, metric="sqeuclidean")
                    scoresdav[ep] = davies_bouldin_score(X, model.labels_)
        sns.lineplot(x=eps, y=scores)
        plt.title('checks')
        plt.show()
        return scoressil, scoresdav
        
        
eps = [0.001,0.018,0.01,0.02,0.03,0.04]

check_DBSCAN_params(eps=eps, X=X, epsc=True)
        
#Selecting hyperparameters for dbscan using 'imported' function

'''
def get_kdist_plot(X=None, k=None, radius_nbrs=1.0):
    nbrs = NearestNeighbors(n_neighbors=k, radius=radius_nbrs).fit(X)
    # For each point, compute distances to its k-nearest neighbors
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:, k-1]
    # Plot the sorted K-nearest neighbor distance for each point in the dataset
    plt.figure(figsize=(8,8))
    plt.plot(distances)
    plt.xlabel('Points/Objects in the dataset', fontsize=12)
    plt.ylabel('Sorted {}-nearest neighbor distance'.format(k), fontsize=12)
    plt.grid(True, linestyle="--", color='black', alpha=0.4)
    plt.show()
    plt.close()
k = 2 * X.shape[-1] - 1 # k=2*{dim(dataset)} - 1
get_kdist_plot(X=x, k=k)

'''


import seaborn as sns
import matplotlib.pyplot as plt


def DBSCAN_check_eps_only(X,eps,min_samples):
    scoressilhouette={}
    scoresdavid = {}
    for ep in eps:
        model =  DBSCAN(eps=ep, min_samples = min_samples)
        result= model.fit(X)
        #predictions = result.predict(X)
        if len(set(model.labels_))>1:
            print(ep, set(model.labels_))
            scoressilhouette[ep] = silhouette_score(X, model.labels_, metric="sqeuclidean")
            scoresdavid[ep] = davies_bouldin_score(X, model.labels_)
    print('Silhouette score: ', scoressilhouette)
    print('\n')
    print('Davies score: ', scoresdavid)

DBSCAN_check_eps_only(X, eps, min_samples=50)

# best model with eps=0.02 and min_samples=50
# important not only to look for sore result, also the nummer of clusters and especially
# the number of values per cluster have to make sense regarding your business case !

modeldb = DBSCAN(eps=0.02, min_samples=50)
resultdb = modeldb.fit(X)

modeldb.labels_

data['DBSCAN_labels'] = modeldb.labels_

data['DBSCAN_labels'].value_counts()


# 3rd model: 
    
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(X, method='ward'))

# problem here is my computer processor, not capable of running through all the different n options for clustering

# lets try elbow method again
model = AgglomerativeClustering()
visualizer = KElbowVisualizer(estimator = model, k = (2,10))
visualizer.fit(X)
visualizer.poof()

# result --> 5 clusters
import collections

a = [1,1,1,1,2,2,2,2,3,3,4,5,5]
counter=collections.Counter(a)

def metrics_withoutlabel_check(X,labels, model):
    print(silhouette_score(X, labels, metric="sqeuclidean"))
    print(davies_bouldin_score(X, labels))
    counter=collections.Counter(labels)
    print(counter)
    plt.bar(counter.keys(), counter.values())
    plt.title(model)
    
def DBSCAN_parameter_tuning(X,eps,min_samples,min_clusters):
    scoressilhouette={}
    scoresdavid = {}
    for mini in min_samples:
        print(mini)
        for ep in eps:
            model =  DBSCAN(eps=ep, min_samples = mini)
            model.fit(X)
            #predictions = result.predict(X)
            if len(set(model.labels_))>=min_clusters:
                print(ep, set(model.labels_))
                scoressilhouette[(ep,mini)] = silhouette_score(X, model.labels_, metric="sqeuclidean")
                scoresdavid[(ep,mini)] = davies_bouldin_score(X, model.labels_)
    print('Silhouette score: ', scoressilhouette)
    print('\n')
    print('Davies score: ', scoresdavid)
    maxsil = max(scoressilhouette, key=scoressilhouette.get)
    maxdav = max(scoresdavid, key=scoresdavid.get)
    print('\n')
    print(f'Silhouete max: {maxsil}')
    print(f'Davies max: {maxdav}')
    print('\n')
    print('Check frequency distribution of Silhouette model')
    model =  DBSCAN(eps=maxsil[0], min_samples = maxsil[1])
    model.fit(X)
    counter=collections.Counter(model.labels_)
    plt.bar(counter.keys(), counter.values())
    plt.title(model)
    plt.show()

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(5)
ac.fit(X)
ac_labels = ac.labels_
set(ac.labels_)

# lets check the results of the different models to find which one has performed the best
# and gives us a realistic amount of cars per cluster

# AGGC
metrics_withoutlabel_check(X=X, labels=ac.labels_)

'''
result AGGC:
    0.5141706395789164
    0.6832946082772218
    Counter({2: 3525, 0: 2919, 1: 270, 4: 206, 3: 80})
'''
'KMEANS'

metrics_withoutlabel_check(X=X, labels=km.labels_)
'''
result KMEANS:
    0.5473970599608073
    0.6760170141060052
    Counter({4: 3452, 0: 2898, 3: 329, 1: 237, 2: 84})
'''
# DBSCAN

eps = [0.001,0.018,0.01,0.02,0.03,0.04]
min_samples = [20,40,60,80,100,200,300]

check_DBSCAN_params(X=X,eps=eps,epsc=True,minc=True)
# spyder is giving me an internal error here running this function, lets try the simpler one
   

DBSCAN_check_eps_only(X=X, eps=eps, min_samples=min_samples)

modeldb = DBSCAN(eps=0.02, min_samples=50)
resultdb = modeldb.fit(X)
counter=collections.Counter(modeldb.labels_)
metrics_withoutlabel_check(X=X, labels=modeldb.labels_, model='DBSCAN')

'''
result DBSCAN:
    0.7586192920363823
    2.4735209334411663
    Counter({0: 6322, -1: 261, 2: 184, 1: 172, 3: 61})

'''
# DBSCAN has best metrics result but lets have our nusiness case in mind, we want to define
# are clusters not too big so we can position drivers at certain points in th city where
# they wont have to drive too far to their next client, if we have one huge one with other
# smaller ones far away, that does help us define certain outlier areas we can position drivers
# at but still leaves the rest of the city as one big cluster with out any clustering inside 
# the city, and therfor no information where to position the drivers

# either we therfor take AGGC or KMEANS, using a little more equally distributed model
# or option 2: take DBSCAN and try to find new clusters deviding the one big clusters
# or option 3: when we see that the one big cluster the requests are so equally distributed
# there is no need for further clustering, we can simply devide the city in certain
# areas which each will be assigned to a certain number of drivers

# though I believe this equal distribution will change if we group by week day and especially
# time of the day, therefor reclustering/option 2 would be the most fitted slution
# for this business case


# lets first expor the clean data and create a map to have an overview of the
# cluster situation

### !!! Best model !!! DBSCAN in this scenario


data['KMEANS_label'] = km.labels_
data['DBSCAN_label'] = modeldb.labels_
data['AGGC_label'] = ac.labels_

data.to_csv(r'C:\Users\leube\Ironhack\Ironprojects\Module_3\Unsupervised-learning-uber-location-forecast\clean_prepped_data.csv')


# resolve problem, lets properly encode day so we will now later one which number corresponds
# to which day during the week

data['Day'] = tst['Day']
del data['Day']

data['Day'] = data['Date/Time'].dt.dayofweek


# now lets continue with option 2 mentioned earlier and lets recluster the one big cluster

# lets import clean data used for mapping in jupyter



datacl = pd.read_csv(r"C:\Users\leube\Ironhack\Ironprojects\Module_3\Unsupervised-learning-uber-location-forecast\clean_prepped_data.csv")

# lets see if we can make further clusters for the oe big cluster in the middle of manhatten

datacl.columns

del datacl['Unnamed: 0']
del datacl['DBSCAN_labels']
del datacl['Date/Time']

filter0 = datacl[datacl['DBSCAN_label']==0]

filter0.shape

# lets try the models

Xcl = filter0[['Lat','Lon']]

# check elbow method to recieve ultimate number of clusters

model = KMeans()
visualizer = KElbowVisualizer(estimator = model, k = (2,10))
visualizer.fit(Xcl)
visualizer.poof()

# result n=4

km = KMeans(n_clusters=4)
result = km.fit(Xcl)
predictk5 = result.predict(Xcl)

km.labels_

metrics_withoutlabel_check(X=Xcl, labels=km.labels_,model=km)

'''
0.5826489959149139
0.847883425674516
Counter({1: 2523, 0: 2201, 3: 863, 2: 735})

'''

# prety decent model, but we want even more evenly distrivuted clusters, lets check if that
# is doable with remainingly decent scores


km = KMeans(n_clusters=6)
result = km.fit(Xcl)
predictk5 = result.predict(Xcl)

km.labels_

metrics_withoutlabel_check(X=Xcl, labels=km.labels_,model=km)

'''
0.5339621236355986
0.8537084729691052
Counter({1: 1914, 5: 1659, 2: 1301, 4: 702, 3: 375, 0: 371})

'''

# similar scores, a little less clear defined clusters but we have what we want -->
# more equally devided clusters

km = KMeans(n_clusters=8)
result = km.fit(Xcl)
predictk5 = result.predict(Xcl)

km.labels_

metrics_withoutlabel_check(X=Xcl, labels=km.labels_,model=km)

'''
0.5221967106251256
0.9058231438396913
Counter({7: 1783, 4: 1265, 3: 888, 1: 832, 2: 540, 0: 387, 5: 370, 6: 257})

'''

# lets take these results and check if the clusters make sense when mapping, 
# scoes here no significant difference and lets try to see if they give the results that we want

filter0['KMEANS_label'] = km.labels_

filter0['KMEANS_label'].value_counts()

filter0.to_csv(r'C:\Users\leube\Ironhack\Ironprojects\Module_3\Unsupervised-learning-uber-location-forecast\central_cluster_clustering.csv')


# find cluster centers

km.cluster_centers_

# make example on how DBSCAN hyper parameter tuning funvtion works

data
X = data[['Lat','Lon']]
eps = [0.001,0.018,0.01,0.02,0.03,0.04]
min_samples = [20,40,60,80,100,200,300]

DBSCAN_parameter_tuning(X=X, eps=eps, min_samples=min_samples, min_clusters=5)















