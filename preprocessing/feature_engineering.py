from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(
                df[feature]
            )
        return df_transformed

class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        return df_transformed

class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0, 1)):
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        return df_transformed

class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.encoder = OneHotEncoder(sparse_output=False, drop="first")

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        return df_transformed

class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._strategy.apply_transformation(df)


if __name__ == "__main__":
    data_path = 'C:/Users/karti/Unknown/ARCAP/Tasks/Completed/04-08-25/real_estate/data/no_outliers_data.csv'
    df = pd.read_csv(data_path)

    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # numerical_columns = [col for col in numerical_columns if col != 'SalePrice']

    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    log_transformer = FeatureEngineer(LogTransformation(features=numerical_columns))
    df_log_transformed = log_transformer.apply_feature_engineering(df)
    
    onehot_encoder = FeatureEngineer(OneHotEncoding(features=categorical_columns))
    df_onehot_encoded = onehot_encoder.apply_feature_engineering(df_log_transformed)
    
    df_onehot_encoded.to_csv('data/cleaned_data.csv', index=False)