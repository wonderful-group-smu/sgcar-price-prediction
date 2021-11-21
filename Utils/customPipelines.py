import numpy as np
import pandas as pd
from collinearity import SelectNonCollinear
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import (GradientBoostingRegressor, IsolationForest,
                              RandomForestRegressor, StackingRegressor)
from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import (FeatureUnion, Pipeline, _fit_transform_one,
                              _transform_one)
from sklearn.preprocessing import (LabelBinarizer, LabelEncoder, MinMaxScaler,
                                   OneHotEncoder, PolynomialFeatures,
                                   StandardScaler)
from sklearn.svm import SVC
from .pipelineComponents import *
RANDOM_STATE = 2021

def get_pipeline_clean_encode_only(categorical_features=[], numerical_features=[]):
  # Clean data with categorical encoding
  categorical_pipeline = Pipeline([
    ('feature_selector', FeatureSelector(feature_names=categorical_features)), 
    ('categorical_imputer', CustomImputer(impute_type='categorical')),
    ('encoder', CustomOneHotEncoder())
  ])
  numerical_pipeline = Pipeline([
    ('feature_selector', FeatureSelector(feature_names=numerical_features))
  ])
  preprocessed_pipeline = PandasFeatureUnion([
    ('categorical_pipeline', categorical_pipeline), 
    ('numerical_pipeline', numerical_pipeline),
  ])
  return preprocessed_pipeline


def get_pipeline_clean_encode_impute(categorical_features=[], numerical_features=[]):
  categorical_pipeline = Pipeline([
    ('feature_selector', FeatureSelector(feature_names=categorical_features)),
    ('categorical_imputer', CustomImputer(impute_type='categorical')),
    ('encoder', CustomOneHotEncoder()),
  ])
  numerical_pipeline = Pipeline([
    ('feature_selector', FeatureSelector(feature_names=numerical_features)),
    ('numerical_imputer', CustomImputer(impute_type='numerical'))
  ])
  preprocessed_pipeline = PandasFeatureUnion([
    ('categorical_pipeline', categorical_pipeline), 
    ('numerical_pipeline', numerical_pipeline), 
  ])
  return preprocessed_pipeline


def get_pipeline_clean_encode_outlier_impute(categorical_features=[], numerical_features=[], outlier_numerical_features=[]):
  categorical_pipeline = Pipeline([
    ('feature_selector', FeatureSelector(feature_names=categorical_features)),
    ('categorical_imputer', CustomImputer(impute_type='categorical')),
    ('encoder', CustomOneHotEncoder()),
  ])
  numerical_pipeline = Pipeline([
    ('feature_selector', FeatureSelector(feature_names=numerical_features)),
    ('numerical_imputer', CustomImputer(impute_type='numerical'))
  ])
  outlier_numerical_pipeline = Pipeline([
    ('feature_selector', FeatureSelector(feature_names=outlier_numerical_features)),
    ("outlier_handler", OutlierHandler(outlier_algorithm="IsolationForest")), # Mark outlier values as NaN
    ('numerical_imputer', CustomImputer(impute_type='numerical'))
  ])

  preprocessed_pipeline = PandasFeatureUnion([
    ('categorical_pipeline', categorical_pipeline),
    ('numerical_pipeline', numerical_pipeline), 
    ('outlier_numerical_pipeline', outlier_numerical_pipeline)
  ])
  return preprocessed_pipeline