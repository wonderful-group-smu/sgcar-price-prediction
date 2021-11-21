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

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
RANDOM_STATE = 2021


class FeatureSelector(BaseEstimator, TransformerMixin):
  """
  Simple transformer to select features. Fits well with sklearn Pipeline.
  """
  def __init__(self, feature_names):
    if not isinstance(feature_names, list):
      raise ValueError("Incorrect format - expected list")
    self.feature_names = feature_names

  def fit(self, X, y=None):
    return self
  
  def transform(self, X):
    if not isinstance(X, pd.DataFrame):
      raise ValueError("Incorrect format - expected pd.DataFrame")
    return X[self.feature_names]

class OutlierHandler(BaseEstimator, TransformerMixin):
  """
  Custom algorithm to handle outliers
  """
  def __init__(self, outlier_algorithm):
    if outlier_algorithm not in ['EllipticEnvelope' ,'IsolationForest']:
      raise ValueError("Incorrect outlier algorithm - expected 'EllipticEnvelope' or 'IsolationForest'")
    self.outlier_algorithm = outlier_algorithm

  def fit(self, X, y=None):
    if self.outlier_algorithm == 'EllipticEnvelope':
        outlier_model = EllipticEnvelope(random_state=RANDOM_STATE)
    elif self.outlier_algorithm == "IsolationForest":
        outlier_model = IsolationForest(random_state=RANDOM_STATE, contamination=0.01) # Reduce contamination to 0.01, else the code won't work
    self.outlier_models = {} # Create multiple outlier model for each feature
    for col in X.columns:
        vals = X[col].dropna() # Drop NA for model fit
        self.outlier_models[col] = outlier_model.fit(vals.values.reshape(-1,1))
    return self
  
  def transform(self, X):
    if not isinstance(X, pd.DataFrame):
      raise ValueError("Incorrect format")
    for i, col in enumerate(X.columns):
        vals = X[col].fillna(0) # Fill na so that prediction can go through (doesn't accept NaN)
        pred = self.outlier_models[col].predict(vals.values.reshape(-1,1))
        outlier_loc = np.where(pred == -1)[0]
        X.iloc[outlier_loc, i] = np.NaN
        # print(f"# of Outliers for {col} - {len(outlier_loc)}, {round(len(outlier_loc)/len(X) * 100,2)} %")
    return X

class CustomImputer(BaseEstimator, TransformerMixin):
  """
  Given only CATEGORICAL or NUMERICAL columns, impute based on certain rules
  NOTE: FeatureSelector should be done before this layer
  """
  def __init__(self, impute_type='numerical'):
    if impute_type not in ['categorical', 'numerical']:
      raise ValueError("Invalid impute type - expected 'cateogrical' or 'numerical'")
    self.impute_type = impute_type

  def fit(self, X, y=None):
    if not isinstance(X, pd.DataFrame):
      raise ValueError("Incorrect format")
    if self.impute_type == 'numerical':
      self.imputer = SimpleImputer(strategy='median')
    elif self.impute_type == 'categorical':
      self.imputer = SimpleImputer(strategy='most_frequent')

    self.imputer.fit(X)
    return self
  
  def transform(self, X):
    transformed_dataframe = pd.DataFrame(self.imputer.transform(X), columns=X.columns, index=X.index)
    return transformed_dataframe

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
  """
  Given only CATEGORICAL columns, outputs one-hot encoded data
  NOTE: FeatureSelector and imputation should be done before this layer
  """
  def fit(self, X, y=None):
    if not isinstance(X, pd.DataFrame):
      raise ValueError("Incorrect format")
    # We would need to multiple binarizers because each only takes in series not dataframe
    self.binarizers = {}
    for column_name in X.columns:
      self.binarizers[column_name] = LabelBinarizer()
      self.binarizers[column_name].fit(X[column_name])
    return self
  
  def transform(self, X):
    if len(self.binarizers) != len(X.columns):
      raise ValueError("Mismatched number of categorical columns")
    
    transformed_df = pd.DataFrame()
    # Categorical encoding without dropping any column e.g: 3 classes -> 4 columns
    for column_name in X.columns:
      classes = self.binarizers[column_name].classes_
      transformed_cols = self.binarizers[column_name].transform(X[column_name])
      if len(classes) == 2:
        transformed_cols_df =  pd.DataFrame(transformed_cols, columns=[f"{column_name}_{classes[1]}"], index=X.index)
      else:
        transformed_cols_df =  pd.DataFrame(transformed_cols, columns=[f"{column_name}_{class_}" for class_ in classes], index=X.index)
      transformed_df = pd.concat([transformed_df, transformed_cols_df], axis=1)
         
    return transformed_df

class PandasSelectNonCollinear(BaseEstimator, TransformerMixin):
  """
  Only selects columns that are non collinear
  Takes in Pandas DataFrame and outputs Pandas DataFrame
  https://www.yourdatateacher.com/2021/06/28/a-python-library-to-remove-collinearity/
  """
  def __init__(self, correlation_threshold=None, scoring=None):
    self.correlation_threshold = correlation_threshold
    self.scoring = scoring
    
  def fit(self, X, y=None):
    if not isinstance(X, pd.DataFrame):
      raise ValueError("Incorrect format")
    self.columns = []
    self.selector = SelectNonCollinear(
      correlation_threshold=self.correlation_threshold,
      scoring=self.scoring
    )
    self.selector.fit(X.to_numpy())
    support = self.selector.get_support()
    for idx in range(len(X.columns)):
      if support[idx] == True:
        self.columns.append(X.columns[idx])
    return self

  def transform(self, X):
    if not isinstance(X, pd.DataFrame):
      raise ValueError("Incorrect format")

    return X[self.columns]

class PandasStandardScaler(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    if not isinstance(X, pd.DataFrame):
      raise ValueError("Incorrect format")
    self.columns = X.columns
    self.scaler = StandardScaler()
    self.scaler.fit(X)
    return self

  def transform(self, X):
    if not isinstance(X, pd.DataFrame):
      raise ValueError("Incorrect format")
    transformed_df = pd.DataFrame(self.scaler.transform(X), columns=self.columns, index=X.index)
    return transformed_df

class PandasMinMaxScaler(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    if not isinstance(X, pd.DataFrame):
      raise ValueError("Incorrect format")
    self.columns = X.columns
    self.scaler = MinMaxScaler()
    self.scaler.fit(X)
    return self

  def transform(self, X):
    if not isinstance(X, pd.DataFrame):
      raise ValueError("Incorrect format")
    transformed_df = pd.DataFrame(self.scaler.transform(X), columns=self.columns, index=X.index)
    return transformed_df

class PCAFeatureReducer(BaseEstimator, TransformerMixin):
  def __init__(self, n_components=None):
    self.n_components = n_components
    
  def fit(self, X, y=None):
    if not isinstance(X, pd.DataFrame):
      raise ValueError("Incorrect format")
    self.columns = []
    self.pca = PCA(n_components=self.n_components, random_state=RANDOM_STATE)
    self.pca.fit(X)
    self.num_components = self.pca.components_.shape[0]
    self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
    self.components_ = self.pca.components_
    return self

  def transform(self, X):
    if not isinstance(X, pd.DataFrame):
      raise ValueError("Incorrect format")
    transformed_df = pd.DataFrame(self.pca.transform(X), columns=[f"Component #{i+1}" for i in range(self.num_components)], index=X.index)
    return transformed_df

class PandasFeatureUnion(FeatureUnion):
  """https://zablo.net/blog/post/pandas-dataframe-in-scikit-learn-feature-union/index.html"""
  def fit_transform(self, X, y=None, **fit_params):
    self._validate_transformers()
    result = Parallel(n_jobs=self.n_jobs)(
      delayed(_fit_transform_one)(
        transformer=trans,
        X=X,
        y=y,
        weight=weight,
        **fit_params)
      for name, trans, weight in self._iter())

    if not result:
      # All transformers are None
      return np.zeros((X.shape[0], 0))
    Xs, transformers = zip(*result)
    self._update_transformer_list(transformers)
    if any(sparse.issparse(f) for f in Xs):
      Xs = sparse.hstack(Xs).tocsr()
    else:
      Xs = self.merge_dataframes_by_column(Xs)
    return Xs

  def merge_dataframes_by_column(self, Xs):
    return pd.concat(Xs, axis="columns", copy=False)

  def transform(self, X):
    Xs = Parallel(n_jobs=self.n_jobs)(
      delayed(_transform_one)(
        transformer=trans,
        X=X,
        y=None,
        weight=weight)
      for name, trans, weight in self._iter())
    if not Xs:
      # All transformers are None
      return np.zeros((X.shape[0], 0))
    if any(sparse.issparse(f) for f in Xs):
      Xs = sparse.hstack(Xs).tocsr()
    else:
      Xs = self.merge_dataframes_by_column(Xs)
    return Xs
  
class RandomForestFeatureSelector(BaseEstimator, TransformerMixin):
  """
  Selects features from random forest feature importance.
  https://scikit-learn.org/stable/modules/feature_selection.html#select-from-model
  Note that high cardinality features may be biased.
  Threshold determines how many features to select. It can be a metric such as median or raw thresholds.
  """
  def __init__(self, threshold="0.5*median"):
    self.selector = SelectFromModel(RandomForestRegressor(), threshold=threshold)
    self.threshold = threshold

  def fit(self, X, y=None):
    if not isinstance(X, pd.DataFrame):
      raise ValueError("Incorrect format")
    self.selector.fit(X, y)
    self.columns = X.columns[self.selector.get_support()]
    return self

  def transform(self, X):
    if not isinstance(X, pd.DataFrame):
      raise ValueError("Incorrect format")
    return X[self.columns]