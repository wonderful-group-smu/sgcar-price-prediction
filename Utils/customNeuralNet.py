from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.base import RegressorMixin
from numpy.random import seed
import tensorflow 
from sklearn.model_selection import train_test_split

RANDOM_STATE = 2021 
seed(RANDOM_STATE)
tensorflow.random.set_seed(RANDOM_STATE)
tensorflow.keras.backend.set_floatx('float64')

class CustomNN(BaseEstimator, RegressorMixin):

  def __init__(self, learning_rate=0.001, hidden_layer_sizes=(64,32,16,1), epochs=500, batch_size=128, verbose=0, validation_frac=0.15):
    
    self.batch_size=batch_size
    self.epochs=epochs
    self.hidden_layer_sizes=hidden_layer_sizes
    self.learning_rate=learning_rate

    self.verbose=verbose
    self.validation_frac=validation_frac
    self._build_model()

  def _build_model(self):
    def get_model():
      model = Sequential()
      model.add(Flatten())
      for idx, hidden_layer_size in enumerate(self.hidden_layer_sizes):
        if idx == len(self.hidden_layer_sizes) - 1:
          model.add(Dense(hidden_layer_size, kernel_initializer='normal')) # Last layer no activation
        else:
          model.add(Dense(hidden_layer_size, kernel_initializer='normal', activation='relu'))
        
      opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
      model.compile(loss='mean_squared_error', optimizer=opt)
      return model
    self.model = KerasRegressor(build_fn=get_model, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

  def fit(self, X, y=None):
    calls = [EarlyStopping(monitor='loss', patience=10)]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=self.validation_frac, random_state=42)
    self.model.fit(X_train, y_train, validation_data=(X_valid, y_valid), callbacks=calls)
    return self
  
  def predict(self, X):
    return self.model.predict(X)