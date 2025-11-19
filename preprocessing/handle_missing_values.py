from abc import ABC, abstractmethod
import pandas as pd

class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis=0, thresh=None):
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        return df_cleaned

class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, method="mean", fill_value=None):
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()
    
        numeric_columns = df_cleaned.select_dtypes(include="number").columns
        if self.method == "mean":
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].mean()
            )
        elif self.method == "median":
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].median()
            )
        elif self.method == "mode":
            for column in numeric_columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace=True)
        elif self.method == "constant":
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(self.fill_value)
            
        categorical_columns = df_cleaned.select_dtypes(include="object").columns
        for column in categorical_columns:
            if self.method == "mode":
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace=True)
            elif self.method == "constant":
                df_cleaned[column].fillna(self.fill_value, inplace=True)
            else:
                df_cleaned[column].fillna('missing', inplace=True)
        
        return df_cleaned

class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._strategy.handle(df)

if __name__ == "__main__":
    data_path = 'C:/Users/karti/Unknown/ARCAP/Tasks/Completed/04-08-25/real_estate/data/AmesHousing.csv'
    df = pd.read_csv(data_path)
    print(f"Original data shape: {df.shape}")
    
    missing_value_handler = MissingValueHandler(DropMissingValuesStrategy(axis=0, thresh=3))
    df_cleaned = missing_value_handler.handle_missing_values(df)
    
    missing_value_handler.set_strategy(FillMissingValuesStrategy(method='mean'))
    df_filled = missing_value_handler.handle_missing_values(df_cleaned)

    print(f"Final data shape: {df_filled.shape}")
    print(f"Missing values in final data: {df_filled.isnull().sum().sum()}")
    
    df_filled.to_csv('data/no_missing_values_data.csv', index=False)
