from abc import ABC, abstractmethod
import pandas as pd
import streamlit as st
from core.interfaces import PreprocessingStep

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

class MissingValuesStep(PreprocessingStep):
    @property
    def name(self) -> str:
        return "Missing Values Handling"

    @property
    def description(self) -> str:
        return "Handle missing values by dropping rows or filling with specific values."

    def render_ui(self, df: pd.DataFrame) -> dict:
        st.subheader("Missing Values Handling")
        
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            st.success("✅ No missing values found in the dataset!")
            return None

        st.write(f"**Total missing values:** {missing_count}")
        
        # Calculate and display missing percentages
        missing_series = df.isnull().sum()
        missing_series = missing_series[missing_series > 0]
        missing_percentages = (missing_series / len(df)) * 100
        
        missing_df = pd.DataFrame({
            "Missing Values": missing_series,
            "Percentage (%)": missing_percentages
        }).sort_values("Percentage (%)", ascending=False)
        
        st.write("**Columns with missing values:**")
        st.dataframe(missing_df, use_container_width=True)

        params = {}

        # 1. Drop columns by threshold
        st.write("### 1. Drop Columns by Threshold")
        threshold_percent = st.slider(
            "Drop columns with missing values > X%",
            min_value=0,
            max_value=100,
            value=100,
            step=5,
            help="Columns with more than this percentage of missing values will be dropped.",
            key="mv_col_threshold"
        )
        params["col_threshold"] = threshold_percent
        
        cols_to_drop_threshold = missing_percentages[missing_percentages > threshold_percent].index.tolist()
        if cols_to_drop_threshold:
            st.warning(f"⚠️ {len(cols_to_drop_threshold)} columns will be dropped based on threshold: {', '.join(cols_to_drop_threshold)}")

        # 2. Manual Column Drop
        st.write("### 2. Manually Drop Columns")
        manual_cols_to_drop = st.multiselect(
            "Select specific columns to drop",
            options=df.columns,
            default=[],
            key="mv_manual_drop"
        )
        params["manual_cols_to_drop"] = manual_cols_to_drop

        # 3. Handle Remaining Missing Values
        st.write("### 3. Handle Remaining Missing Values")
        st.info("This strategy will be applied to any remaining missing values after the column drops above.")
        
        handling_method = st.radio(
            "Select Handling Method",
            ["Drop Rows", "Fill Missing Values"],
            horizontal=True,
            key="mv_method"
        )
        params["method"] = handling_method
        
        if handling_method == "Drop Rows":
            drop_option = st.selectbox(
                "Drop Option",
                ["Drop rows with any missing values", 
                 "Drop rows with all missing values"],
                key="mv_drop_option"
            )
            params["drop_option"] = drop_option
        
        else:  # Fill Missing Values
            fill_method = st.selectbox(
                "Fill Method",
                ["Mean", "Median", "Mode", "Constant Value"],
                key="mv_fill_method"
            )
            params["fill_method"] = fill_method
            
            if fill_method == "Constant Value":
                fill_value = st.text_input("Enter constant value to fill", key="mv_fill_value")
                params["fill_value"] = fill_value
        
        return params

    def execute(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        if params is None:
            return df
            
        df_processed = df.copy()
        
        # 1. Drop columns by threshold
        col_threshold = params.get("col_threshold", 100)
        if col_threshold < 100:
            missing_percentages = (df_processed.isnull().sum() / len(df_processed)) * 100
            cols_to_drop = missing_percentages[missing_percentages > col_threshold].index.tolist()
            if cols_to_drop:
                df_processed.drop(columns=cols_to_drop, inplace=True)
        
        # 2. Manual Column Drop
        manual_cols = params.get("manual_cols_to_drop", [])
        if manual_cols:
            # Only drop if they exist (might have been dropped by threshold already)
            cols_to_drop = [c for c in manual_cols if c in df_processed.columns]
            if cols_to_drop:
                df_processed.drop(columns=cols_to_drop, inplace=True)
        
        # 3. Handle Remaining
        method = params.get("method")
        
        if method == "Drop Rows":
            drop_option = params.get("drop_option")
            axis = 0
            threshold = None
            
            if drop_option == "Drop rows with all missing values":
                # thresh requires at least ONE non-NA value to KEEP. 
                # So if we want to drop rows with ALL missing, we need thresh=1 (keep if at least 1 is present)
                # Wait, dropna(how='all') is equivalent to thresh=1? No.
                # dropna(how='all') drops if ALL are NA.
                # dropna(how='any') drops if ANY is NA.
                
                # The existing strategy uses 'thresh'.
                # DropMissingValuesStrategy uses dropna(thresh=self.thresh)
                # If thresh is None, it defaults to 'any' behavior if how is not specified?
                # Let's check DropMissingValuesStrategy again.
                # It calls df.dropna(axis=self.axis, thresh=self.thresh)
                # If thresh is None, it drops any NA? No, dropna default is how='any'.
                
                # Let's just use the strategy as intended or fix it if needed.
                # But for "Drop rows with all missing values", we can pass thresh=1 (keep row if it has at least 1 non-NA)
                # Actually, let's look at how it was implemented before:
                # if drop_option == "Drop rows with all missing values": threshold = df.shape[1]
                # That means keep only if it has ALL values present? No.
                # thresh=N means "require that many non-NA values".
                # If threshold = df.shape[1], it means "require ALL columns to be non-NA". That is "how='any'".
                
                # I will implement it directly here for clarity or reuse strategy carefully.
                # Let's reuse the strategy but be mindful of what it does.
                pass

            # Re-instantiate strategy based on processed dataframe
            if drop_option == "Drop rows with all missing values":
                 # We want to drop rows where ALL columns are missing.
                 # So we keep rows with at least 1 non-missing value.
                 threshold = 1 
            else:
                 # Drop rows with ANY missing values
                 # We keep rows with ALL non-missing values.
                 threshold = df_processed.shape[1]
            
            strategy = DropMissingValuesStrategy(axis=axis, thresh=threshold)
            handler = MissingValueHandler(strategy)
            return handler.handle_missing_values(df_processed)
            
        else:
            fill_method = params.get("fill_method")
            fill_value = params.get("fill_value")
            
            # Convert fill value if possible
            if fill_value:
                try:
                    fill_value = float(fill_value)
                except ValueError:
                    pass
            
            method_map = {
                "Mean": "mean",
                "Median": "median",
                "Mode": "mode",
                "Constant Value": "constant"
            }
            
            strategy = FillMissingValuesStrategy(
                method=method_map[fill_method],
                fill_value=fill_value
            )
            handler = MissingValueHandler(strategy)
            return handler.handle_missing_values(df_processed)
