from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import os
import joblib
os.makedirs('models', exist_ok=True)

class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        pass

class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        pipeline = Pipeline(
            [
                # ("scaler", StandardScaler()), 
                ("model", LinearRegression()), 
            ]
        )

        pipeline.fit(X_train, y_train) 
        return pipeline

class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        return self._strategy.build_and_train_model(X_train, y_train)

if __name__ == "__main__":
    data_path_X_train = 'C:/Users/karti/Unknown/ARCAP/Tasks/Completed/04-08-25/real_estate/data/train/X_train_data.csv'
    data_path_y_train = 'C:/Users/karti/Unknown/ARCAP/Tasks/Completed/04-08-25/real_estate/data/train/y_train_target.csv'
    
    df_Xtrain = pd.read_csv(data_path_X_train)
    df_ytrain = pd.read_csv(data_path_y_train).squeeze()
    
    model_builder = ModelBuilder(LinearRegressionStrategy())
    trained_model = model_builder.build_model(df_Xtrain, df_ytrain)

    joblib.dump(trained_model, 'models/trained_model1.pkl')
    print("Model saved successfully!")