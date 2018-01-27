dm_mobility_task
==============================


# Introduction

In this project we analyse mobility data gathered via smartphones and model it with different
clustering techniques. This project was done during the Data Mining course of
the University of Vienna.

# Notebooks

The following notebooks are a selection of tasks that we had to submit or find
particularly interesting:

- [basic visualization](https://github.com/univie-datamining-team3/assignment2/blob/master/notebooks/3.0-lm-visualisation-trip-segments-with-distance-metric.ipynb)
- [base-Line KMeans - all trips](https://github.com/univie-datamining-team3/assignment2/blob/master/notebooks/4.1-lm-distances-feature-engineered-n2-all-trips.ipynb)
- [base-line KMeans with distances calculated via x,y,z axis - only scripted trips](https://github.com/univie-datamining-team3/assignment2/blob/master/notebooks/4.0-lm-base-line-kmeans-xyz-scripted.ipynb)
- [clustering with feature engineered distances](https://github.com/univie-datamining-team3/assignment2/blob/master/notebooks/4.1-lm-distances-feature-engineered-n2-all-trips.ipynb)
- [clustering with dtw and L2 norm](https://github.com/univie-datamining-team3/assignment2/blob/master/notebooks/4.3-lm-kmeans-dtw-scripted.ipynb)
- [clustering with automated feature engineering with tsfresh](https://github.com/univie-datamining-team3/assignment2/blob/master/notebooks/4.4-lm-kmeans-euclidean-tsfresh-all.ipynb)
- [bayesian tsne optimzation](https://github.com/univie-datamining-team3/assignment2/blob/master/notebooks/5.0-lm-optimization-tsne.ipynb)


If you want to run the notebooks yourself you have to complete the following steps.


# How To

## Environment Setup

In order to run the project you have to install the required packages. We
recommend [conda](https://www.anaconda.com/download/) for Windows/Linux/Mac or
[virtualenv](https://pypi.python.org/pypi/virtualenv) for Linux/Mac OS for creating
a new python environment. Note that we only tested the code with python version 3.5.

If you run Linux/Mac OS you can install all dependencies via:

```
pip install -r requirements.txt
```

For Windows you should use conda to install the following packages:
```
conda create --name test_environment python=3.5 Cython pandas scikit-learn numpy scipy seaborn psutil hdbscan
```

The other packages can be installed via pip.
Note that if you want to run the notebook on Bayesian optimization of t-SNE, you have to install the repository "coranking" directly from its repository:
```
pip install git+https://github.com/samueljackson92/coranking.git
```

Now before you can run the [notebooks](https://github.com/univie-datamining-team3/assignment2/blob/master/notebooks) you have to create a .env file in the cloned repository folder assignment2\\.env and add the URL and authentication for accessing the data in the following way:

```
# Key for mobility data
KEY_LUKAS=??????
KEY_RAPHAEL=??????
KEY_MORITZ=??????

# Authentication
URL =??????
LOGINNAME=??????
LOGINPASSWORD =??????
```

After that you can download all the data by starting the script from the source folder via:

```
python data/make_dataset.py --download True --preprocess True --distance_metric euclidean
```


This will take a while and depends on your internet connection as a bunch of data
is downloaded and preprocessed. The preprocessing takes about 20 minutes.
After that you have several new files in data/preprocessed and data/raw.

Now you are good to go to execute the different jupyter [notebooks](https://github.com/univie-datamining-team3/assignment2/blob/master/notebooks).



Project Organization
------------

    ├── LICENSE
    ├── README.md          
    ├── data
    │   ├── preprocessed   <- data that has been preprocessed.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Documentation as pdf
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── make_dataset.py
    │   │   ├── data_utils.py
    │   │   └── preprocessing.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to for clustering algorithms
    │   │   │                 
    │   │   ├── cluster.py
    │   │   ├── elki_main.py
    │   │   ├── evaluate_distance_metrics.py
    │   │   └── dimensionality_reduction
    │   │       ├── BayesianTSNEOptimizer.py
    │   │       └── TSNEModel.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │    └── visualize.py
    │   │
    │   └── utils  <- Utility scripts and helper functions
    │       └── utilities.py



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
