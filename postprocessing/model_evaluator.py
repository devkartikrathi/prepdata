from abc import ABC, abstractmethod

import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score

class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        pass

class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {"Mean Squared Error": mse, "R-Squared": r2}
        return metrics


class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        self._strategy = strategy

    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        return self._strategy.evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    model = joblib.load('models/trained_model1.pkl')

    data_path_X_test = 'C:/Users/karti/Unknown/ARCAP/Tasks/Completed/04-08-25/real_estate/data/test/X_test_data.csv'
    data_path_y_test = 'C:/Users/karti/Unknown/ARCAP/Tasks/Completed/04-08-25/real_estate/data/test/y_test_target.csv'
    
    df_Xtest = pd.read_csv(data_path_X_test)
    df_ytest = pd.read_csv(data_path_y_test).squeeze()

    model_evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
    evaluation_metrics = model_evaluator.evaluate(model, df_Xtest, df_ytest)
    print(evaluation_metrics)
