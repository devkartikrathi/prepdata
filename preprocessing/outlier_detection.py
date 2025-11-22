from abc import ABC, abstractmethod
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from core.interfaces import PreprocessingStep

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

class OutlierDetectionStep(PreprocessingStep):
    @property
    def name(self) -> str:
        return "Outlier Detection"

    @property
    def description(self) -> str:
        return "Detect and handle outliers using Z-Score or IQR methods."

    def render_ui(self, df: pd.DataFrame) -> dict:
        st.subheader("Outlier Detection & Treatment")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            st.warning("⚠️ No numerical columns found for outlier detection")
            return None
        
        selected_cols = st.multiselect(
            "Select columns for outlier detection",
            numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))],
            key="outlier_cols"
        )
        
        if len(selected_cols) == 0:
            return None
            
        detection_method = st.radio(
            "Detection Method",
            ["Z-Score", "IQR (Interquartile Range)"],
            horizontal=True,
            key="outlier_method"
        )
        
        threshold = 3.0
        if detection_method == "Z-Score":
            threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.5, key="outlier_threshold")
            detector = OutlierDetector(ZScoreOutlierDetection(threshold=threshold))
        else:
            detector = OutlierDetector(IQROutlierDetection())
        
        df_numeric = df[selected_cols]
        outliers = detector.detect_outliers(df_numeric)
        outlier_count = outliers.any(axis=1).sum()
        
        st.write(f"**Number of rows with outliers:** {outlier_count}")
        st.write(f"**Percentage of outliers:** {(outlier_count / len(df)) * 100:.2f}%")
        
        # Visualization
        if len(selected_cols) > 0:
            num_plots = min(len(selected_cols), 4)
            cols = st.columns(2)
            for idx, col_name in enumerate(selected_cols[:num_plots]):
                with cols[idx % 2]:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.boxplot(y=df[col_name], ax=ax)
                    ax.set_title(f"Boxplot: {col_name}")
                    st.pyplot(fig)
                    plt.close()
        
        treatment_method = st.radio(
            "Treatment Method",
            ["Remove Outliers", "Cap Outliers (Winsorization)"],
            horizontal=True,
            key="outlier_treatment"
        )
        
        return {
            "selected_cols": selected_cols,
            "detection_method": detection_method,
            "threshold": threshold,
            "treatment_method": treatment_method
        }

    def execute(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        if params is None:
            return df
            
        selected_cols = params.get("selected_cols")
        detection_method = params.get("detection_method")
        threshold = params.get("threshold")
        treatment_method = params.get("treatment_method")
        
        if not selected_cols:
            return df
            
        if detection_method == "Z-Score":
            detector = OutlierDetector(ZScoreOutlierDetection(threshold=threshold))
        else:
            detector = OutlierDetector(IQROutlierDetection())
            
        df_numeric = df[selected_cols]
        
        if treatment_method == "Remove Outliers":
            df_numeric_cleaned = detector.handle_outliers(df_numeric, method="remove")
            df_categorical = df.drop(columns=selected_cols)
            # Keep only rows that survived outlier removal
            df_categorical_cleaned = df_categorical.loc[df_numeric_cleaned.index]
            df_cleaned = pd.concat([df_numeric_cleaned, df_categorical_cleaned], axis=1)
        else:
            df_numeric_cleaned = detector.handle_outliers(df_numeric, method="cap")
            df_categorical = df.drop(columns=selected_cols)
            df_cleaned = pd.concat([df_numeric_cleaned, df_categorical], axis=1)
        
        # Reorder columns to match original
        return df_cleaned[df.columns]
