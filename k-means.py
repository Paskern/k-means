# Imports
from os import access
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy.random.mtrand import normal
import pandas as pd
import numpy as np
import copy
from mpl_toolkits.mplot3d import Axes3D

# Generate dataframe with the size of (100,3) with elements between [0,100)
df = pd.DataFrame(np.random.randint(0,100,size=(100,3)), columns=list('xyz'))

# Define number of centroids and initialize random centroids
k = 3
centroids = pd.DataFrame(np.random.randint(0,100,size=(k,3)), columns=list('xyz'))

# Enter df in visual figure
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df['x'], df['y'], df['z'], color='k')

# Enter centroids in visual figure
colors='rgbcmy'
for i in range(k):
    ax.scatter(centroids['x'][i], centroids['y'][i], centroids['z'][i], color=colors[i%6])

plt.show()

print(df.head)

def calculate_distance(df, centroids):
    
    for i in range(k):
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids['x'][i]) ** 2
                + (df['y'] - centroids['y'][i]) ** 2
                + (df['z'] - centroids['z'][i]) ** 2
            )
        )

    # Find the closest centroid and color the right color
    distance_cols = ['distance_from_{}'.format(i) for i in range(k)]
    df['closest_centroid'] = df.loc[:, distance_cols].idxmin(axis=1)
    df['closest_centroid'] = df['closest_centroid'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest_centroid'].map(lambda x: colors[x%6])

    return df

# Call functions and print info in console
calculate_distance(df, centroids)
print(df.head())

# Prepare for visualization
fig2 = plt.figure()
ax2 = Axes3D(fig2)
ax2.scatter(df['x'], df['y'], df['z'], color=df['color'])
for i in range(k):
    ax2.scatter(centroids['x'][i], centroids['y'][i], centroids['z'][i], color=colors[i%6], edgecolor='k', s=70)

# Show visuals
plt.show()

### Updating new location of centroids ###

# Copy old centroid
centroids_old = copy.deepcopy(centroids)

def calculate_new_centroid(l):
    for i in range(k):
        centroids['x'][i] = np.mean(df[df['closest_centroid'] == i]['x'])
        centroids['y'][i] = np.mean(df[df['closest_centroid'] == i]['y'])
        centroids['z'][i] = np.mean(df[df['closest_centroid'] == i]['z'])
    return l


centroids = calculate_new_centroid(centroids)
print(centroids_old)
print(centroids)

fig3 = plt.figure()
ax3 = Axes3D(fig3)

ax3.scatter(df['x'], df['y'], df['z'], color=df['color'])
for i in range(k):
    ax3.scatter(centroids_old['x'][i], centroids_old['y'][i], centroids_old['z'][i], color=colors[i%6], edgecolor='k', s=60, alpha=0.5)
    ax3.scatter(centroids['x'][i], centroids['y'][i], centroids['z'][i], color=colors[i%6], edgecolor='k', s=80)

plt.show()

