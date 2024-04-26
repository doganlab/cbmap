#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CBMAP: Clustering-based Manifold Approximation and Projection for Dimensionality Reduction

"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA


def cluster(X, method, params):
    if method == "kmeans":
        nc = params["n_clusters"]
        kmeans = MiniBatchKMeans(n_clusters=nc, batch_size = 2048, random_state=params["random_state"]).fit(X)
        centers = kmeans.cluster_centers_
        y = kmeans.labels_

    return centers, y


class CBMAP(BaseEstimator):
    """ Clustering-based Manifold Approximation and Projection (CBMAP)
    
    Parameters
    ----------
    n_components : int, default=2
        Dimension of the projected space
    max_iter : int, default=500
        Maximum number of iterations of the CBMAP algorithm for a single run.
    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    clustering_method: str, default = "kmeans"


    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Stores the position of the dataset in the embedding space.
    labels : cluster labels
    CH     : cluster centers in high-dimensional space
    sigmaH : sigma value for high-dimensional space
    CL     : cluster centers in low-dimensional space
    sigmaL : sigma value for low-dimensional space
    
    
    Examples
    --------
    >>> from sklearn import datasets
    >>> import matplotlib.pyplot as plt
    >>> n_samples = 1000
    >>> S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)
    >>> params = {"n_clusters" : 20,"random_state": 0}
    >>> cbmapObj = CBMAP(params, clustering_method = "kmeans")
    >>> S_cbmap = cbmapObj.fit_transform(S_points)
    >>> plt.scatter(S_cbmap[:,0], S_cbmap[:,1], c=S_color)

    """
    
    def __init__(self, params, max_iter=500, clustering_method = "kmeans", center_init = "PCA", n_components=2):
        self.params = params
        self.max_iter = max_iter
        self.clustering_method = clustering_method
        self.n_components = n_components
        self.center_init = center_init
        self.labels = [] 
        self.CL = []
        self.sigmaH = []
        self.sigmaL = []

        if self.n_components < 2:
            raise ValueError("The number of output dimensions must be at least 2.")
        if self.clustering_method not in ("kmeans"):
            raise ValueError("The clustering_method must be stated as kmeans")
        if self.center_init not in ("PCA", "random"):
            raise ValueError("The center_init parameter either must be PCA or random.")

    def _membership(self, X, C, sigma):
        d = cdist(X,C,metric='euclidean')
        u = np.exp(-d**2/(2*sigma**2)) 
        
        if np.isnan(d).any():
            raise ValueError("Distance between at least one cluster center to a data point is close to zero. Try using a smaller number of clusters, k.")
            exit
        
        return u, d


    def _data_embedding(self, UH, CH, CL, l, epochs, dim, labels, sigmaL, sigmaH):
        """ This function embeds the data points in low-dimensional space
            by using the membership function UH computed in high-dimensional space and the PCA-embedded
            or randomly-initialized low-dimensional centers CL.
        """
        
        size = UH.shape
        
        Y = np.random.randn(size[0], dim) + CL[labels,:]
        
        
        for k in range(epochs):      
            UL, d = self._membership(Y, CL, sigmaL)
            
            F = np.linalg.norm(UH - UL) 

            v = np.zeros((size[0]))
            m = np.zeros((size[0], dim))
            mg = np.zeros((size[0], dim))
            ng = np.zeros((size[0]))

            # adam optimization
            U = (UL - UH) / F * -UL / sigmaL**2
            mg = np.einsum('ij,ijk->ik', U,  Y[:,None] - CL)  
            ng = np.sum(mg**2, 1) 
            if k == 0:
                m = 0.001*mg
                v = 0.1*ng
            else:
                v = 0.9*v + 0.1*ng
                m = 0.999*m + 0.001*mg
    
            m = m / (1 - 0.999**(k+1))    
            v = v / (1 - 0.9**(k+1))    
            
            # update data positions
            Y = Y - m/np.sqrt(np.reshape(v, (size[0], 1)) + 1e-8)*l
            
            # update the cluster centers and sigma value
            for j in range(CL.shape[0]):
                indx = np.where(labels == j)[0]
                CL[j,:] = np.mean(Y[indx,:],0)
                
            CL = (CL - np.mean(CL,0)) / np.std(CL,0)
            dCL = cdist(CL,CL)
            sigmaL = np.mean(np.median(dCL, 0))
            

        return Y, CL, sigmaL

    
    def _data_transform(self, X, UH, epochs, dim, l):
        
        size = UH.shape
        
        Y = np.random.randn(X.shape[0], dim) 
                
        for k in range(epochs):      
            
            UL, d = self._membership(Y, self.CL, self.sigmaL)
            
            F = np.linalg.norm(UH - UL) 

            v = np.zeros((size[0]))
            m = np.zeros((size[0], dim))
            mg = np.zeros((size[0], dim))
            ng = np.zeros((size[0]))

            # adam optimization
            U = (UL - UH) / F * -UL / self.sigmaL**2

            mg = np.einsum('ij,ijk->ik', U,  Y[:,None] - self.CL)  
            ng = np.sum(mg**2, 1) 
            if k == 0:
                m = 0.001*mg
                v = 0.1*ng
            else:
                v = 0.9*v + 0.1*ng
                m = 0.999*m + 0.001*mg
    
            m = m / (1 - 0.999**(k+1))    
            v = v / (1 - 0.9**(k+1))    
            
            # update positions
            Y = Y - m/np.sqrt(np.reshape(v, (size[0], 1)) + 1e-8)*l
        
        return Y


    def transform(self, X):
        CH = self.CH
        sigmaH = self.sigmaH
        dCH = cdist(X,CH)
        UH = np.exp(-dCH**2/(2*sigmaH**2)) 
         
        Y = self._data_transform(X, UH, self.max_iter, self.n_components, 0.9)
        
        return Y

    def fit(self, X, y=None, init=None):
        """
        Compute the position of the points in the embedding space.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # parameter will be validated in `fit_transform` call
        self.fit_transform(X, init=init)
        return self
            
    
    def fit_transform(self, X, y=None, init=None):
        """
        Fit the data from `X`, and returns the embedded coordinates.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            X transformed in the new space.
        """

        CH, labels = cluster(X, self.clustering_method, self.params)
        self.labels = labels
        dCH = cdist(X,CH)
        sigmaH = np.mean(np.median(dCH, 0))
        UH, d = self._membership(X, CH, sigmaH) 
        
        if self.center_init == "PCA":
            CL = PCA(n_components=self.n_components).fit_transform(CH)
        elif self.center_init == "random":
            Y = np.random.randn(X.shape[0], self.n_components)
            CL = np.zeros((CH.shape[0], self.n_components))
            # find the cluster centers
            for j in range(CL.shape[0]):
                indx = np.where(labels == j)[0]
                CL[j,:] = np.mean(Y[indx,:],0)

        CL = (CL - np.mean(CL,0)) / np.std(CL,0)
        dCL = cdist(CL,CL)
        sigmaL = np.mean(np.median(dCL, 0))
                  
        self.embedding_, CL, sigmaL = self._data_embedding(UH, CH, CL, 0.99, self.max_iter, self.n_components, labels, sigmaL, sigmaH)
    
        self.CH = CH
        self.sigmaH = sigmaH
        self.CL = CL
        self.sigmaL = sigmaL    
    
        return self.embedding_

