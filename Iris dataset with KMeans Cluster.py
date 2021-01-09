# Species Segmentation with Clusrter Analysis

# The Iris flower dataset is one of the most popular ones for machine learning. You can read a lot about it online and have probably already heard of it: https://en.wikipedia.org/wiki/Iris_flower_data_set
#
# We didn't want to use it in the lectures, but believe that it would be very interesting for you to try it out (and maybe read about it on your own).
#
# There are 4 features: sepal length, sepal width, petal length, and petal width.
#
# Start by creating 2 clusters. Then standardize the data and try again. Does it make a difference?
#
# Use the Elbow rule to determine how many clusters are there.

## Import the relevant libraries
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
## Load the data
#  Load the  iris_dataset.csv into data
data = pd.read_csv('iris_dataset.csv')
# check the data and null values
print(data.head)
print(data.describe(include='all'))
print(data.isnull().sum())
print(data.columns.values)

## Plot the data
# For this exercise, try to cluster the iris flowers by the shape of their sepal.
# Use the 'sepal_length' and 'sepal_width' variables.
# create a scatter plot based on two corresponding features (sepal_length and sepal_width; OR petal_length and petal_width)
plt.scatter(data['sepal_length'],data['sepal_width'])
# name your axes
plt.xlabel('Length of Sepal')
plt.ylabel('Width of Sepal')
plt.show()
## Clustering (unscaled data)
# create a variable which will contain the data for the clustering
x = data.copy()
# create a k-means object with 2 clusters
kmeans = KMeans(2)
# fit the data
kmeans.fit(x)
# create a copy of data, so we can see the clusters next to the original data
clusters = data.copy()
# predict the cluster for each observation
clusters['Clusters_pred'] = kmeans.fit_predict(x)
print(clusters)
# create a scatter plot based on two corresponding features (sepal_length and sepal_width; OR petal_length and petal_width)
plt.scatter(clusters['sepal_length'],clusters['sepal_width'],c=clusters['Clusters_pred'],cmap='rainbow')
plt.show()
## Standardize the variables
# import some preprocessing module
from sklearn import preprocessing
# scale the data for better results
X_scaled =preprocessing.scale(data)
print(X_scaled)
## Clustering (scaled data)
# create a k-means object with 2 clusters
kmeans_new = KMeans(3)
# fit the data
kmeans_new.fit(X_scaled)
# create a copy of data, so we can see the clusters next to the original data
clusters_scaled = data.copy()
# predict the cluster for each observation
clusters_scaled['clusters_pred']=kmeans_new.fit_predict(X_scaled)
print(clusters_scaled)
# create a scatter plot based on two corresponding features (sepal_length and sepal_width; OR petal_length and petal_width)
plt.scatter(clusters_scaled['sepal_length'],clusters_scaled['sepal_width'],c=clusters_scaled['clusters_pred'],cmap='rainbow')
plt.title(' Kmeans Cluster with scaled data')
plt.xlabel('Length of Sepal')
plt.ylabel('Width of Sepal')
plt.show()

# Looks like the two solutions are identical. That is because the original features have very similar scales to start with!
## Take Advantage of the Elbow Method
wcss = []
# 'cl_num' is a that keeps track the highest number of clusters we want to use the WCSS method for.
# We have it set at 10 right now, but it is completely arbitrary.
cl_num =10
for i in range(1,cl_num):
    kmeans_new= KMeans(i)
    kmeans_new.fit(X_scaled)
    wcss_inte = kmeans_new.inertia_
    wcss.append(kmeans_new.inertia_)
print(wcss)
# It seems like 2 or 3-cluster solutions are the best.
number_cluster = range(1,cl_num)
plt.plot(number_cluster,wcss)
plt.title('The Elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within Sum of Squares')
plt.show()
# Based on the Elbow Curve, plot several graphs with the appropriate amounts of clusters you believe would best fit the data.

