"""Loading, splitting and normalizing the Boston Housing data."""
import numpy as np

DATA_PATH = r"C:\Users\dylan\OneDrive\Documents\4A\SYDE 572\Assignments\Assignment 2\assn2 material\assn2 material\housing.data"

# Column order of housing.data (see housing.names). The handout numbers the features from 1
# (RM is the 6th feature, DIS the 8th), so feature number n is Python index n - 1.
FEATURE_NAMES = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
                 "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
FEATURE_DESCRIPTIONS = [
    "per capita crime rate",
    "proportion of residential land zoned for lots over 25,000 sq.ft.",
    "proportion of non-retail business acres",
    "Charles River dummy (1 if tract bounds river)",
    "nitric oxides concentration (parts per 10 million)",
    "average number of rooms per dwelling",
    "proportion of owner-occupied units built before 1940",
    "weighted distance to five Boston employment centers",
    "index of accessibility to radial highways",
    "full-value property-tax rate per $10,000",
    "pupil-teacher ratio",
    "1000(Bk - 0.63)^2",
    "% lower status of the population",
]

N_TRAIN = 100          # the first 100 towns are the training set
FEATURE_1D = 1         # Xn(:,2) / "2nd feature", counting from 1 as in MATLAB -> ZN


def load_housing(path=DATA_PATH):
    """Return the 506 x 13 feature matrix X and the 506 targets y (MEDV)."""
    data = np.loadtxt(path)
    return data[:, :13], data[:, 13]


def split(X, y, n_train=N_TRAIN):
    """The first n_train rows are training data and the rest are test data."""
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def fit_normalizer(X_train):
    """
    Mean and standard deviation of each feature, from the training rows only.

    CHAS is 0 for all of the first 100 towns, so its training standard deviation is 0.
    A standard deviation of 0 is replaced by 1 so the division is defined. The feature is
    then constant (0) in training and receives no weight.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def normalize(X, mean, std):
    """z-score each feature with the training statistics: z = (x - mean) / std."""
    return (X - mean) / std


def load_normalized(n_train=N_TRAIN):
    """Train/test split with the features normalized using training statistics."""
    X, y = load_housing()
    X_tr, y_tr, X_te, y_te = split(X, y, n_train)
    mean, std = fit_normalizer(X_tr)
    return normalize(X_tr, mean, std), y_tr, normalize(X_te, mean, std), y_te, mean, std
