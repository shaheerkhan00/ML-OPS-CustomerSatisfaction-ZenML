import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression



class Model(ABC):
    """
    Abstract class for all Models
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        Args:
        x_train, y_train: training data
        Retruns:None
        """
        pass
    
class LinearRegressionModel(Model):
    """
    Linear Regression Model
    """
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train,**kwargs):
        """
        Trains the Linear Regression model
        Args:
        x_train, y_train: training data
        Retruns:None
        """
        
        try:
            logging.info("Training Linear Regression model")
            model= LinearRegression(**kwargs)
            model.fit(X_train,y_train)
            return model
            
        except Exception as e:
            logging.error(f"Error in training Linear Regression model: {e}")
            raise e