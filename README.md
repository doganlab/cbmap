# cbmap
CBMAP: Clustering-Based Manifold Approximation and Projection for Dimensionality Reduction
Dogan, B. (2024). CBMAP: Clustering-based manifold approximation and projection for dimensionality reduction. arXiv preprint arXiv:2404.17940.

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
import json
import numpy as np

#load the mammoth dataset
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
f2.close()

colorIndex = -1
nColorPoints = 0

for i in range(len(S_color)):
    if (i >= nColorPoints):
        colorIndex += 1
        nColorPoints += labels[colorIndex]
    S_color[i] = colorIndex


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

3. Fashion MNIST dataset

In this example we wiill split the Fashion MNIST dataset into training and test datasets. We will
first learn the manifold with the training dataset and then project the unseen test data.

```{python}
import cbmap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load Fashion-MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]**2))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]**2))

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

S_points = np.concatenate((X_train, X_test))
S_color = np.concatenate((y_train, y_test))

X_train, X_test, y_train, y_test = train_test_split(S_points, S_color, test_size=0.25, random_state=42)

params = {"n_clusters" : 40, "random_state": 0}
cbmapObj = cbmap.CBMAP(params, clustering_method = "kmeans")
X_train_projected_cbmap = cbmapObj.fit_transform(X_train)
plt.scatter(X_train_projected_cbmap [:,0], X_train_projected_cbmap [:,1], c=y_train, cmap = plt.cm.rainbow, s=2, alpha=0.8)
plt.title("Training set")
plt.show()

X_test_projected_cbmap = cbmapObj.transform(X_test)
plt.scatter(X_test_projected_cbmap [:,0], X_test_projected_cbmap [:,1], c=y_test, cmap = plt.cm.rainbow, s=2, alpha=0.8)
plt.title("Test set")
plt.show()
```

2D embedding for the training data:

![resim](https://github.com/doganlab/cbmap/assets/26445624/0297cc54-0a06-4af6-b8e4-49f0240fc124)

2D embedding for the test data:

![resim](https://github.com/doganlab/cbmap/assets/26445624/89c7abba-4520-4b2b-8788-2d6e13d3a37d)


