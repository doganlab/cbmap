# cbmap
CBMAP: Clustering-based Manifold Approximation and Projection for Dimensionality Reduction
Installation

cbmap can be installed as follows:

    $ pip install cbmap

Also the development verison of cbmap can be installed from master branch of Git repository:

    $ pip install git+https://github.com/doganlab/cbmap

Examples
--------
In this example, the S-curve dataset is used for dimensionality reduction.

![resim](https://github.com/doganlab/cbmap/assets/26445624/d0fbf7e7-a757-482b-baae-585d98de5521)

The two-dimensional projection of the S-curve datasets can be obtained as follows:

```{python}
from sklearn import datasets
import matplotlib.pyplot as plt
n_samples = 1000
S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)
params = {"n_clusters" : 20,"random_state": 0}
cbmapObj = cbmap.CBMAP(params, clustering_method = "kmeans")
S_cbmap = cbmapObj.fit_transform(S_points)
plt.scatter(S_cbmap[:,0], S_cbmap[:,1], c=S_color)
```

The output should be something like this:

![resim](https://github.com/doganlab/cbmap/assets/26445624/56d94380-5b60-4739-b8a0-3a291557c069)

