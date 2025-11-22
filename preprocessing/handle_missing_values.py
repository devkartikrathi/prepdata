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

class AdvancedMissingValueStrategy(MissingValueHandlingStrategy):
    def __init__(self, config: dict):
        """
        config: dict mapping column names to strategy dictionaries.
        Example:
        {
            "age": {"method": "mean"},
            "income": {"method": "median"},
            "category": {"method": "mode"},
            "description": {"method": "constant", "fill_value": "Unknown"},
            "outlier_col": {"method": "drop_rows"}
        }
        """
        self.config = config

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()
        
        # 1. Handle "drop_rows" first to avoid processing rows that will be dropped
        rows_to_drop_mask = pd.Series(False, index=df_cleaned.index)
        
        for col, strategy in self.config.items():
            if col not in df_cleaned.columns:
                continue
                
            if strategy["method"] == "drop_rows":
                # Mark rows where this column is null for dropping
                rows_to_drop_mask |= df_cleaned[col].isnull()
        
        if rows_to_drop_mask.any():
            df_cleaned = df_cleaned[~rows_to_drop_mask]
            
        # 2. Handle imputations
        for col, strategy in self.config.items():
            if col not in df_cleaned.columns:
                continue
                
            method = strategy["method"]
            
            if method == "drop_rows":
                continue # Already handled
                
            if df_cleaned[col].isnull().sum() == 0:
                continue # Nothing to fill
                
            if method == "mean":
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    fill_val = df_cleaned[col].mean()
                    df_cleaned[col] = df_cleaned[col].fillna(fill_val)
            elif method == "median":
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    fill_val = df_cleaned[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(fill_val)
            elif method == "mode":
                if not df_cleaned[col].mode().empty:
                    fill_val = df_cleaned[col].mode().iloc[0]
                    df_cleaned[col] = df_cleaned[col].fillna(fill_val)
            elif method == "constant":
                fill_val = strategy.get("fill_value")
                if fill_val is not None:
                    # Try to convert to numeric if column is numeric
                    if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                        try:
                            fill_val = float(fill_val)
                        except (ValueError, TypeError):
                            pass # Keep as string if conversion fails (though it might fail insertion)
                    df_cleaned[col] = df_cleaned[col].fillna(fill_val)
                    
        return df_cleaned

class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._strategy.handle(df)

class MissingValuesStep(PreprocessingStep):
    @property
    def name(self) -> str:
        return "Missing Values Handling"

    @property
    def description(self) -> str:
        return "Handle missing values with granular control over numeric and categorical columns."

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
        st.write("### 1. Drop Columns")
        with st.expander("Configure Column Dropping", expanded=True):
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
            
            manual_cols_to_drop = st.multiselect(
                "Select specific columns to drop manually",
                options=df.columns,
                default=[],
                key="mv_manual_drop"
            )
            params["manual_cols_to_drop"] = manual_cols_to_drop
            
            all_drop_cols = list(set(cols_to_drop_threshold + manual_cols_to_drop))
            if all_drop_cols:
                st.warning(f"⚠️ {len(all_drop_cols)} columns will be dropped: {', '.join(all_drop_cols)}")

        # Identify remaining columns with missing values
        remaining_missing_cols = [c for c in missing_series.index if c not in all_drop_cols]
        
        if not remaining_missing_cols:
            st.info("All missing values will be handled by dropping columns.")
            return params

        st.write("### 2. Imputation Strategies")
        st.info("Define how to handle missing values for remaining columns.")

        # Global Strategies
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Numeric Columns")
            numeric_strategy = st.selectbox(
                "Global Strategy",
                ["Mean", "Median", "Mode", "Constant", "Drop Rows", "Ignore"],
                key="mv_global_numeric"
            )
            params["global_numeric"] = numeric_strategy
            
            if numeric_strategy == "Constant":
                params["global_numeric_val"] = st.text_input("Value", "0", key="mv_global_num_val")

        with col2:
            st.markdown("#### Categorical Columns")
            categorical_strategy = st.selectbox(
                "Global Strategy",
                ["Mode", "Constant", "Drop Rows", "Ignore"],
                key="mv_global_cat"
            )
            params["global_categorical"] = categorical_strategy
            
            if categorical_strategy == "Constant":
                params["global_cat_val"] = st.text_input("Value", "Missing", key="mv_global_cat_val")

        # Individual Overrides
        st.markdown("#### Individual Column Overrides")
        st.info("Create rules to apply specific strategies to groups of columns.")
        
        if "mv_override_groups_count" not in st.session_state:
            st.session_state.mv_override_groups_count = 1

        def add_group():
            st.session_state.mv_override_groups_count += 1
            
        def reset_groups():
            st.session_state.mv_override_groups_count = 1

        overrides = {}
        
        # Container for groups
        for i in range(st.session_state.mv_override_groups_count):
            with st.expander(f"Override Rule #{i+1}", expanded=True):
                # Filter out columns that are already selected in previous groups? 
                # For simplicity, we show all remaining columns, but user should be careful.
                # Or we can try to filter.
                
                # Let's just show all remaining_missing_cols to avoid complex state management of "used columns"
                # The last rule defined for a column will effectively win if we process in order, 
                # but the UI might be confusing. 
                # Let's try to filter if possible, but standard multiselect doesn't support dynamic options easily 
                # without reruns.
                
                cols = st.multiselect(
                    f"Select columns for Rule #{i+1}",
                    options=remaining_missing_cols,
                    key=f"mv_group_cols_{i}"
                )
                
                if cols:
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        # Show superset of strategies
                        strat_options = ["Mean", "Median", "Mode", "Constant", "Drop Rows"]
                        method = st.selectbox(
                            f"Strategy for Rule #{i+1}", 
                            strat_options, 
                            key=f"mv_group_strat_{i}"
                        )
                    
                    val = None
                    if method == "Constant":
                        with c2:
                            val = st.text_input(f"Value for Rule #{i+1}", key=f"mv_group_val_{i}")
                    
                    # Apply to all selected columns
                    for col in cols:
                        overrides[col] = {"method": method.lower().replace(" ", "_"), "fill_value": val}

        col_add, col_reset = st.columns([1, 5])
        with col_add:
            st.button("➕ Add Rule", on_click=add_group)
        with col_reset:
            st.button("🔄 Reset Rules", on_click=reset_groups)
            
        params["overrides"] = overrides

        return params

    def execute(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        if params is None:
            return df
            
        df_processed = df.copy()
        
        # 1. Drop columns
        col_threshold = params.get("col_threshold", 100)
        manual_cols = params.get("manual_cols_to_drop", [])
        
        cols_to_drop = []
        
        # Threshold drops
        if col_threshold < 100:
            missing_percentages = (df_processed.isnull().sum() / len(df_processed)) * 100
            cols_to_drop.extend(missing_percentages[missing_percentages > col_threshold].index.tolist())
            
        # Manual drops
        cols_to_drop.extend(manual_cols)
        
        # Deduplicate and drop
        cols_to_drop = list(set(cols_to_drop))
        cols_to_drop = [c for c in cols_to_drop if c in df_processed.columns]
        
        if cols_to_drop:
            df_processed.drop(columns=cols_to_drop, inplace=True)
            
        # 2. Build Configuration for Advanced Strategy
        config = {}
        
        global_num_strat = params.get("global_numeric", "Ignore").lower().replace(" ", "_")
        global_num_val = params.get("global_numeric_val")
        
        global_cat_strat = params.get("global_categorical", "Ignore").lower().replace(" ", "_")
        global_cat_val = params.get("global_cat_val")
        
        overrides = params.get("overrides", {})
        
        # Apply global strategies first
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() == 0:
                continue
                
            if col in overrides:
                config[col] = overrides[col]
            else:
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    if global_num_strat != "ignore":
                        config[col] = {"method": global_num_strat, "fill_value": global_num_val}
                else:
                    if global_cat_strat != "ignore":
                        config[col] = {"method": global_cat_strat, "fill_value": global_cat_val}
                        
        # Execute Strategy
        strategy = AdvancedMissingValueStrategy(config)
        handler = MissingValueHandler(strategy)
        return handler.handle_missing_values(df_processed)
