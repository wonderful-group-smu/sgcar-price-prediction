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

RANDOM_STATE = 2021
def get_clean_car_df(car_data, coe_data):
  """
  Format data types properly.
  Remove rows with unimputable missing columns.
  Approximate missing COE value.
  """

  def get_clean_coe_df():
    """Retrieve COE dataset to approximate empty COE values. Will only be used during preprocessing thus inner function."""
    coe_df = pd.read_csv(coe_data, thousands=',')

    # Format data types from strings to datetime and integers
    coe_df['Announcement Date'] = pd.to_datetime(coe_df['Announcement Date'])
    coe_df['Quota Premium'] = coe_df['Quota Premium'].str.replace('$', '').str.replace(',', '').astype(int)

    # Sort by timeline so that the oldest entries are at the top
    coe_df = coe_df.sort_values(by=['Announcement Date'])

    # Filter only for necessary categories of COE - Cat A & Cat B
    coe_df = coe_df[(coe_df['Category'] == 'Cat A (Cars up to 1600cc and 97kW)')|(coe_df['Category'] == 'Cat B (Cars above 1600cc or 97kW)')]

    return coe_df
  
  ## ----------- ##

  car_df = pd.read_csv(car_data)
  original_len = len(car_df)

  # Drop the unnamed column uesless for analysis
  car_df = car_df.drop(columns=['Unnamed: 0'])

  # Remove rows that have blank variables as they cannot be meaningfully imputed
  car_df = car_df[~car_df['PRICE'].isna()] # not meaningful to impute the target variable
  car_df = car_df[~car_df['REG_DATE'].isna()] # not meaningful to impute the car's age as it is unique

  # Format date from string to pandas datetime form easier for analysis
  car_df['REG_DATE'] = np.where(car_df['REG_DATE'] == 'N.A.', np.NaN, car_df['REG_DATE'])
  car_df['REG_DATE'] = pd.to_datetime(car_df['REG_DATE'])
  car_df['SCRAPE_DATE'] = pd.to_datetime(car_df['SCRAPE_DATE'])
  car_df['REG_DATE_DAYS'] = (car_df['SCRAPE_DATE'] - car_df['REG_DATE']).dt.days

  # Change invalid Transmission values to NaN, we would only include Auto and Manual Transmission types
  car_df['TRANSMISSION'] = np.where((car_df['TRANSMISSION'] != 'Auto') & (car_df['TRANSMISSION'] != 'Manual'), np.NaN, car_df['TRANSMISSION'])

  # Create a new column on whether COE is original or approxmiated to preserve data integrity
  car_df['IS_COE_APPROXIMATED'] = car_df['COE_FROM_SCRAPE_DATE'].isna().astype(bool).astype(str)

  # Get the historical COE prices by categories
  coe_df = get_clean_coe_df()

  # Iterate through each car index to replace NaN prices of COE
  for i in car_df.index:
    if car_df.loc[i,'IS_COE_APPROXIMATED']:
      if car_df.loc[i,'ENGINE_CAPACITY_CC'] <= 1600:
        coe_specific_cc_df = coe_df[coe_df['Category'] == 'Cat A (Cars up to 1600cc and 97kW)']
      else:
        coe_specific_cc_df = coe_df[coe_df['Category'] == 'Cat B (Cars above 1600cc or 97kW)']

      # The latest coe announcement date before the registration date of the car
      coe_date = coe_specific_cc_df[coe_specific_cc_df['Announcement Date'] < car_df.loc[i,'REG_DATE']].loc[:,'Announcement Date'].max()

      # If COE date is NaT, means the registration date was very long ago, then leave it as NaT to be removed afterwards
      if coe_date is not pd.NaT:
        car_df.loc[i,'COE_FROM_SCRAPE_DATE'] = coe_specific_cc_df[coe_specific_cc_df['Announcement Date'] == coe_date].loc[:,'Quota Premium'].iloc[0]

  # Remove rows if COE price cannot be approximated due to the registration being too long ago
  car_df = car_df[~car_df['COE_FROM_SCRAPE_DATE'].isna()]

  # Convert OMV string to integer values
  car_df['OMV'] = car_df['OMV'].str.replace('$', '').str.replace(',', '')
  car_df['OMV'] = pd.to_numeric(car_df['OMV'], errors="ignore")

  print(f"Original size: {original_len} -> Cleaned size: {len(car_df)}")
  return car_df

def get_standard_train_test_split(car_df):
  """
  Test set should not be touched until the end of the project.
  Use cross validation for tuning.
  """
  X = car_df.drop(columns='PRICE')
  y = car_df['PRICE']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
  return X_train, X_test, y_train, y_test
