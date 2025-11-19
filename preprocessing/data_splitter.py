from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split

class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str):
        pass

class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test

class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: DataSplittingStrategy):
        self._strategy = strategy

    def split(self, df: pd.DataFrame, target_column: str):
        return self._strategy.split_data(df, target_column)

if __name__ == "__main__":
    data_path = 'C:/Users/karti/Unknown/ARCAP/Tasks/Completed/04-08-25/real_estate/data/cleaned_data.csv'
    df = pd.read_csv(data_path)
    
    print(f"Original data shape: {df.shape}")

    data_splitter = DataSplitter(SimpleTrainTestSplitStrategy(test_size=0.2, random_state=42))
    X_train, X_test, y_train, y_test = data_splitter.split(df, target_column='SalePrice')

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    X_train.to_csv('data/train/X_train_data.csv', index=False)
    X_test.to_csv('data/test/X_test_data.csv', index=False)
    y_train.to_csv('data/train/y_train_target.csv', index=False)
    y_test.to_csv('data/test/y_test_target.csv', index=False)
