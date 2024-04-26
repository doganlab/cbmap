# cbmap
CBMAP: Clustering-based Manifold Approximation and Projection for Dimensionality Reduction
Installation

cbmap can be installed as follows:

    $ pip install cbmap

Also the development verison of cbmap can be installed from master branch of Git repository:

    $ pip install git+https://github.com/doganlab/cbmap

Examples
--------
from sklearn import datasets
import matplotlib.pyplot as plt
n_samples = 1000
S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)
params = {"n_clusters" : 20,"random_state": 0}
cbmapObj = CBMAP(params, clustering_method = "kmeans")
S_cbmap = cbmapObj.fit_transform(S_points)
plt.scatter(S_cbmap[:,0], S_cbmap[:,1], c=S_color)



