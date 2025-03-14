# cbmap
CBMAP: Clustering-Based Manifold Approximation and Projection for Dimensionality Reduction

Please find the details of the method in the following preprint:

[Dogan, B. (2024). CBMAP: Clustering-based manifold approximation and projection for dimensionality reduction. arXiv preprint arXiv:2404.17940.](https://arxiv.org/abs/2404.17940v2)

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

In this example we will split the Fashion MNIST dataset into training and test datasets. We will
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

4. In this example we will demonstrate how CBMAP (Clustering-Based Manifold Approximation and Projection) can effectively reduce the dimensionality of a complex, synthetic 3D dataset while preserving its structure. The dataset consists of:

Two interleaved half-moon clusters (non-linearly separable).
Two concentric circles (nested structures).
A random Z-dimension is added to transform these 2D structures into a 3D dataset.

We will apply CBMAP to reduce this 3D dataset to 2D and compare the effects of using different initialization strategies (center_init = "PCA" and center_init = "random").

The dataset is created by combining:

A moon-shaped dataset (make_moons) with random height variation (Z-axis noise).
A circle-based dataset (make_circles) also with Z-axis noise.
Both datasets are stacked to form a complex, mixed dataset in three dimensions.

```{python}
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_moons, make_circles
from cbmap import CBMAP
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE

# Create two complex 2D datasets
X_moons, color_moons = make_moons(n_samples=1000, noise=0.05, random_state=42)
X_circles, color_circles = make_circles(n_samples=1000, noise=0.05, factor=0.5, random_state=42)

# Add a random Z-dimension to create a 3D dataset
Z_moons = np.random.randn(X_moons.shape[0]) * 0.1
Z_circles = np.random.randn(X_circles.shape[0]) * 0.1

# Combine into 3D coordinates
X_moons_3D = np.column_stack((X_moons, Z_moons))
X_circles_3D = np.column_stack((X_circles, Z_circles))

# Merge both datasets into a single 3D dataset
X_combined_3D = np.vstack((X_moons_3D, X_circles_3D))
color_combined = np.hstack((color_moons, color_circles))

# Visualize the original 3D data
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_combined_3D[:, 0], X_combined_3D[:, 1], X_combined_3D[:, 2], c=color_combined, cmap='Spectral')
ax.set_title("Original 3D Complex Dataset")
plt.show()

```
![image](https://github.com/user-attachments/assets/be915fb2-3f8c-48d4-af83-e6920c12ce85)

We now apply CBMAP to reduce the dataset from 3D to 2D using PCA-based initialization for cluster centers.

```{python}
# CBMAP with PCA-based initialization for cluster centers.
params = {"n_clusters": 25, "random_state": 42, "center_init": "PCA"}
cbmapObj = CBMAP(params, clustering_method="kmeans")

# Perform dimensionality reduction
X_cbmap_pca = cbmapObj.fit_transform(X_combined_3D)

# Visualize the results
plt.figure(figsize=(6, 6))
plt.scatter(X_cbmap_pca[:, 0], X_cbmap_pca[:, 1], c=color_combined, cmap='Spectral')
plt.title("CBMAP Projection (center_init = 'PCA')")
plt.show()
```
To test the robustness of CBMAP, we now change the initialization strategy to "random". The result should remain nearly identical, proving that CBMAP is not sensitive to initialization strategies.

```{python}
# CBMAP with random initialization for cluster centers.
params = {"n_clusters": 25, "random_state": 42, "center_init": "random"}
cbmapObj = CBMAP(params, clustering_method="kmeans")

# Perform dimensionality reduction
X_cbmap_random = cbmapObj.fit_transform(X_combined_3D)

# Visualize the results
plt.figure(figsize=(6, 6))
plt.scatter(X_cbmap_random[:, 0], X_cbmap_random[:, 1], c=color_combined, cmap='Spectral')
plt.title("CBMAP Projection (center_init = 'random')")
plt.show()
```
We compare the PCA-based and random-initialized projections side by side:

```{python}

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# PCA-based projection
ax[0].scatter(X_cbmap_pca[:, 0], X_cbmap_pca[:, 1], c=color_combined, cmap='Spectral')
ax[0].set_title("CBMAP Projection (PCA Initialization)")

# Random-based projection
ax[1].scatter(X_cbmap_random[:, 0], X_cbmap_random[:, 1], c=color_combined, cmap='Spectral')
ax[1].set_title("CBMAP Projection (Random Initialization)")

plt.show()
```
![image](https://github.com/user-attachments/assets/d38d09fe-8084-4fe3-852a-03e6a1fbe92a)

Finally, let's compare CBMAP with PCA, UMAP and t-SNE!

```{python}
# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_combined_3D)

# Apply UMAP for dimensionality reduction
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_combined_3D)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_combined_3D)

# Plot all results side by side
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

# PCA
ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=color_combined, cmap='Spectral')
ax[0].set_title("PCA Projection")

# UMAP
ax[1].scatter(X_umap[:, 0], X_umap[:, 1], c=color_combined, cmap='Spectral')
ax[1].set_title("UMAP Projection")

# t-SNE
ax[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=color_combined, cmap='Spectral')
ax[2].set_title("t-SNE Projection")
```
Here is the output!

![image](https://github.com/user-attachments/assets/3243b9bf-fb81-4e26-9931-61854346945b)



