# cbmap
CBMAP: Clustering-based Manifold Approximation and Projection for Dimensionality Reduction

Installation
--------

cbmap can be installed as follows:

    $ pip install cbmap

Also the development verison of cbmap can be installed from master branch of Git repository:

    $ pip install git+https://github.com/doganlab/cbmap

Examples
--------
1. The S-curve dataset

![resim](https://github.com/doganlab/cbmap/assets/26445624/d0fbf7e7-a757-482b-baae-585d98de5521)

The two-dimensional projection of the S-curve datasets can be obtained as follows:

```{python}
import cbmap
from sklearn import datasets
import matplotlib.pyplot as plt
n_samples = 1000
S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)
params = {"n_clusters" : 20,"random_state": 0}
cbmapObj = cbmap.CBMAP(params, clustering_method = "kmeans")
S_cbmap = cbmapObj.fit_transform(S_points)
plt.scatter(S_cbmap[:,0], S_cbmap[:,1], c=S_color)
```

The output should be something as follows:

![resim](https://github.com/doganlab/cbmap/assets/26445624/56d94380-5b60-4739-b8a0-3a291557c069)

2. The Mammoth dataset

```{python}
import cbmap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker
import json
import numpy as np

#Then we load the mammoth dataset
# Opening JSON file
f = open('./datasets/mammoth_3d.json')
  # returns JSON object as 
# a dictionary
data = json.load(f)

f.close()

S_points = np.array(data)

f2 = open('./datasets/mammoth_umap.json')
    
labels = json.load(f2)
labels = labels['labels']
S_color = np.zeros(len(labels))

colorIndex = -1
nColorPoints = 0

for i in range(len(S_color)):
    if (i >= nColorPoints):
        colorIndex += 1
        nColorPoints += labels[colorIndex]
    S_color[i] = colorIndex

f2.close()

# Creating the plot
fig = plt.figure(figsize=(30, 20))
# Adding 3d scatter plot
ax = fig.add_subplot(211, projection='3d')
ax.view_init(azim=90, elev=-60)
ax.scatter(S_points[:, 0], S_points[:, 1], S_points[:, 2], c = S_color, s = 1)
ax.grid(False)
ax.set_title('Mammoth')
ax.title.set_size(20)
ax.axis('off')
plt.show()
```
This code reads and plots the Mammoth dataset in 3D space.

![resim](https://github.com/doganlab/cbmap/assets/26445624/33e745a3-678f-4b16-b279-30aae4d8a926)


The two-dimensional projection of the Mammoth datasets can be obtained as follows:

```{python}
params = {"n_clusters" : 40, "random_state": 0}
cbmapObj = cbmap.CBMAP(params, clustering_method = "kmeans")
S_cbmap = cbmapObj.fit_transform(S_points)
plt.scatter(S_cbmap[:,0], S_cbmap[:,1], c=S_color, s=2, alpha=0.8)
```
The output should be something as follows:

![resim](https://github.com/doganlab/cbmap/assets/26445624/0bac4552-ca4f-49ac-add4-d93776b32966)


