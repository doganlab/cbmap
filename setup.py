from setuptools import setup

def readme():
    with open('README.md') as readme_file:
        return readme_file.read()

configuration = {
    'name' : 'cbmap',
    'version': '0.1.0',
    'description' : 'CBMAP: Clustering-based Manifold Approximation and Projection for Dimensionality Reduction',
    'long_description' : readme(),
    'classifiers' : [
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.8',
    ],
    'keywords' : 'Dimensionality Reduction UMAP TriMap PaCMAP',
    'url' : 'https://github.com/doganlab/cbmap',
    'author' : 'Berat Dogan',
    'author_email' : 'berat.dogan@inonu.edu.tr',
    'license' : 'LICENSE.txt',
    'packages' : ['cbmap'],
    'install_requires' : ['scikit-learn >= 0.24',
                          'numpy >= 1.24',
                          'scipy >= 1.10']
    }

setup(**configuration)