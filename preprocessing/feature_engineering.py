from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from core.interfaces import PreprocessingStep

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

class CategoricalEncodingStep(PreprocessingStep):
    @property
    def name(self) -> str:
        return "Categorical Encoding"

    @property
    def description(self) -> str:
        return "Convert categorical variables into numerical format using One-Hot Encoding."

    def render_ui(self, df: pd.DataFrame) -> dict:
        st.subheader("Categorical Encoding")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) == 0:
            st.info("ℹ️ No categorical columns found. Skipping this step.")
            return None
            
        st.write("**Available categorical columns:**")
        st.write(", ".join(categorical_cols))
        
        selected_cols = st.multiselect(
            "Select categorical columns to encode",
            categorical_cols,
            key="encoding_cols"
        )
        
        if len(selected_cols) == 0:
            return None
            
        st.subheader("Preview Before Encoding")
        st.dataframe(df[selected_cols].head(), use_container_width=True)
        
        return {"selected_cols": selected_cols}

    def execute(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        if params is None or not params.get("selected_cols"):
            return df
            
        selected_cols = params.get("selected_cols")
        encoder = FeatureEngineer(OneHotEncoding(features=selected_cols))
        return encoder.apply_feature_engineering(df)

class FeatureEngineeringStep(PreprocessingStep):
    @property
    def name(self) -> str:
        return "Feature Engineering"

    @property
    def description(self) -> str:
        return "Apply transformations like Log Transformation to numerical features."

    def render_ui(self, df: pd.DataFrame) -> dict:
        st.subheader("Feature Engineering")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            st.warning("⚠️ No numerical columns found for feature engineering")
            return None
            
        transformation_type = st.selectbox(
            "Select Transformation",
            ["Log Transformation"],
            key="fe_transform_type"
        )
        
        if transformation_type == "Log Transformation":
            st.write("**Log transformation applies log1p(x) = log(1+x) to handle zero values**")
            
            selected_cols = st.multiselect(
                "Select numerical features for log transformation",
                numeric_cols,
                key="fe_log_cols"
            )
            
            if len(selected_cols) > 0:
                # Check for negative or zero values
                invalid_cols = []
                for col in selected_cols:
                    if (df[col] <= 0).any():
                        invalid_cols.append(col)
                
                if invalid_cols:
                    st.warning(f"⚠️ Columns with non-positive values: {', '.join(invalid_cols)}. "
                              f"Log transformation may not be appropriate.")
                
                st.subheader("Preview Before Transformation")
                st.dataframe(df[selected_cols].head(), use_container_width=True)
                st.write("**Statistics:**")
                st.dataframe(df[selected_cols].describe(), use_container_width=True)
                
                return {
                    "transformation_type": transformation_type,
                    "selected_cols": selected_cols
                }
        
        return None

    def execute(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        if params is None or not params.get("selected_cols"):
            return df
            
        selected_cols = params.get("selected_cols")
        transformation_type = params.get("transformation_type")
        
        if transformation_type == "Log Transformation":
            transformer = FeatureEngineer(LogTransformation(features=selected_cols))
            return transformer.apply_feature_engineering(df)
            
        return df

class FeatureScalingStep(PreprocessingStep):
    @property
    def name(self) -> str:
        return "Feature Scaling"

    @property
    def description(self) -> str:
        return "Scale numerical features using Standard Scaling or Min-Max Scaling."

    def render_ui(self, df: pd.DataFrame) -> dict:
        st.subheader("Feature Scaling")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            st.warning("⚠️ No numerical columns found for scaling")
            return None
            
        scaling_method = st.radio(
            "Scaling Method",
            ["Standard Scaling (Z-score)", "Min-Max Scaling"],
            horizontal=True,
            key="scaling_method"
        )
        
        selected_cols = st.multiselect(
            "Select numerical features to scale",
            numeric_cols,
            key="scaling_cols"
        )
        
        if len(selected_cols) > 0:
            st.subheader("Preview Before Scaling")
            st.dataframe(df[selected_cols].head(), use_container_width=True)
            st.write("**Statistics:**")
            st.dataframe(df[selected_cols].describe(), use_container_width=True)
            
            params = {
                "scaling_method": scaling_method,
                "selected_cols": selected_cols
            }
            
            if scaling_method == "Min-Max Scaling":
                feature_range = st.slider(
                    "Feature Range",
                    min_value=0.0,
                    max_value=1.0,
                    value=(0.0, 1.0),
                    step=0.1,
                    key="scaling_range"
                )
                params["feature_range"] = feature_range
                
            return params
            
        return None

    def execute(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        if params is None or not params.get("selected_cols"):
            return df
            
        selected_cols = params.get("selected_cols")
        scaling_method = params.get("scaling_method")
        
        if scaling_method == "Standard Scaling (Z-score)":
            scaler = FeatureEngineer(StandardScaling(features=selected_cols))
        else:
            feature_range = params.get("feature_range", (0, 1))
            scaler = FeatureEngineer(MinMaxScaling(features=selected_cols, feature_range=feature_range))
        
        return scaler.apply_feature_engineering(df)