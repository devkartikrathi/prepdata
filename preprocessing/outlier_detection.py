from abc import ABC, abstractmethod
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = z_scores > self.threshold
        return outliers

class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
        return outliers

class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        self._strategy = strategy

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._strategy.detect_outliers(df)

    def handle_outliers(self, df: pd.DataFrame, method="remove", **kwargs) -> pd.DataFrame:
        outliers = self.detect_outliers(df)
        if method == "remove":
            df_cleaned = df[(~outliers).all(axis=1)]
        elif method == "cap":
            df_cleaned = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
        else:
            return df
        return df_cleaned

    def visualize_outliers(self, df: pd.DataFrame, features: list):
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()

if __name__ == "__main__":
    data_path = 'C:/Users/karti/Unknown/ARCAP/Tasks/Completed/04-08-25/real_estate/data/no_missing_values_data.csv'
    df = pd.read_csv(data_path)
    print(f"Original data shape: {df.shape}")
    
    df_numeric = df.select_dtypes(include=[np.number])
    df_categorical = df.select_dtypes(exclude=[np.number])
    
    outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
    # outlier_detector = OutlierDetector(IQROutlierDetection())
    # outlier_detector.visualize_outliers(df_numeric_cleaned, features=["SalePrice", "Gr Liv Area"])

    outliers = outlier_detector.detect_outliers(df_numeric)

    df_numeric_cleaned = outlier_detector.handle_outliers(df_numeric, method="remove")
    df_categorical_cleaned = df_categorical.loc[df_numeric_cleaned.index]
    df_combined = pd.concat([df_numeric_cleaned, df_categorical_cleaned], axis=1)

    print(f"Final combined shape: {df_combined.shape}")
    print(f"Missing values in final data: {df_combined.isnull().sum().sum()}")
    
    df_combined.to_csv('data/no_outliers_data.csv', index=False)
