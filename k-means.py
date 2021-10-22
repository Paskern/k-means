# Imports
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy

# User choice of how data is generated
choice_data = input('Choice\n\n1. Generate random points\n2. Generate 3 clusters\n\n')

if choice_data == '1':
    # Generate dataframe with random points: Size=100x3, Values: [0,200)
    df = pd.DataFrame(np.random.randint(0,100,size=(200,3)), columns=list('xyz'))
elif choice_data == '2':
    # Generate dataframe with 3 clusters of data: 
    df = pd.DataFrame(np.random.randint(0,50,size=(100,3)), columns=list('xyz'))
    df = df.append(pd.DataFrame(np.random.randint(35,75,size=(100,3)), columns=list('xyz')))
    df = df.append(pd.DataFrame(np.random.randint(60,100,size=(100,3)), columns=list('xyz')))
else:
    print('Error. Try again!')
    quit()

# User choice of number of centroids and initialize random centroids
k = int(input('\nEnter number of centroids:'))
centroids = pd.DataFrame(np.random.randint(0,100,size=(k,3)), columns=list('xyz'))

# Enter df in visual figure
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1, projection="3d")
ax1.scatter(df['x'], df['y'], df['z'], color='k', s=2)

# Enter centroids in visual figure
colors='rgbcmy'
for i in range(k):
    ax1.scatter(centroids['x'][i], centroids['y'][i], centroids['z'][i], edgecolor='k', color=colors[i%6], s=15)

# Function to calculate distance to centroids
def calculate_distance(df, centroids):
    # Calculate the distance and adding it to df in a new column
    for i in range(k):
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids['x'][i]) ** 2
                + (df['y'] - centroids['y'][i]) ** 2
                + (df['z'] - centroids['z'][i]) ** 2
            )
        )

    # Find the closest centroid and the corresponding color
    distance_cols = ['distance_from_{}'.format(i) for i in range(k)]
    df['closest_centroid'] = df.loc[:, distance_cols].idxmin(axis=1)
    df['closest_centroid'] = df['closest_centroid'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest_centroid'].map(lambda x: colors[x%6])

    return df

# Call function to calculate distance
calculate_distance(df, centroids)

# Enter new df in visual figure
ax2 = fig.add_subplot(2,2,2, projection="3d")
ax2.scatter(df['x'], df['y'], df['z'], color=df['color'], s=2)
for i in range(k):
    ax2.scatter(centroids['x'][i], centroids['y'][i], centroids['z'][i], color=colors[i%6], edgecolor='k', s=15)

# Copy of old centroids
centroids_old = copy.deepcopy(centroids)

# Function to update new location of centroids
def calculate_new_centroid(l):
    for i in range(k):
        centroids['x'][i] = np.mean(df[df['closest_centroid'] == i]['x'])
        centroids['y'][i] = np.mean(df[df['closest_centroid'] == i]['y'])
        centroids['z'][i] = np.mean(df[df['closest_centroid'] == i]['z'])
    return l

# Call function to update location of centroids (1st round)
centroids = calculate_new_centroid(centroids)
print('\n', centroids_old)
print('\n', centroids)

# Call function to update distance to centroids (1st round)
df = calculate_distance(df, centroids)

# Enter new centroid in visual figure
ax3 = fig.add_subplot(2,2,3, projection="3d")
ax3.scatter(df['x'], df['y'], df['z'], color=df['color'], s=2)
for i in range(k):
    ax3.scatter(centroids_old['x'][i], centroids_old['y'][i], centroids_old['z'][i], color=colors[i%6], edgecolor='k', s=15, alpha=0.25)
    ax3.scatter(centroids['x'][i], centroids['y'][i], centroids['z'][i], color=colors[i%6], edgecolor='k', s=15)

# Complete algorithm until no difference in clustering
while True:
    closest_centroid = df['closest_centroid'].copy(deep=True)
    centroids = calculate_new_centroid(centroids)
    df = calculate_distance(df, centroids)

    if closest_centroid.equals(df['closest_centroid']):
        break

# Enter final clusters in visual figure
ax4 = fig.add_subplot(2,2,4, projection="3d")
ax4.scatter(df['x'], df['y'], df['z'], color=df['color'], s=2)
for i in range(k):
    ax4.scatter(centroids['x'][i], centroids['y'][i], centroids['z'][i], color=colors[i%6], edgecolor='k', s=15)

plt.show()