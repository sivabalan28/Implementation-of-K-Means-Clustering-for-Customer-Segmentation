# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Pick customer segment quantity (k)
2. Seed cluster centers with random data points.
3. Assign customers to closest centers.
4. Re-center clusters and repeat until stable.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: ATCHAYA S
RegisterNumber: 212222040021 
*/
```
```
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data

x=data[['Annual Income (k$)', 'Spending Score (1-100)']]
x

plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending score (1-100)')
plt.show()

k=5
kmeans = KMeans(n_clusters=k)
kmeans.fit(x)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("Centroid:")
print(centroids)
print("Labels:")
print(labels)

colors =['r','g','b','c','m']
for i in range(k):
  cluster_points =x[labels == i]
  plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'],color=colors[i], label = f'Cluster {i+1}')
  distances = euclidean_distances(cluster_points,[centroids[i]])
  radius = np.max(distances)
  circle = plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
  
plt.scatter(centroids[:,0], centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('k-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show
```
## Output:
![image](https://github.com/AtchayaSundaramoorthy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393516/bcf18dbe-5191-499c-a39c-ab78500e7751)

![image](https://github.com/AtchayaSundaramoorthy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393516/61e18729-00ae-402c-9b5d-5b9bf5391e04)

![image](https://github.com/AtchayaSundaramoorthy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393516/fe6390b4-ea6d-4a4f-8468-0e5bf21405e6)

![image](https://github.com/AtchayaSundaramoorthy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393516/a99db6d4-93eb-4ddd-96fe-805c24c756a5)

![image](https://github.com/AtchayaSundaramoorthy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393516/44d72b76-f1e7-4133-a88a-aca6bef70989)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
