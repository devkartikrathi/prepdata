import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Import preprocessing modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from preprocessing.handle_missing_values import (
    MissingValueHandler,
    DropMissingValuesStrategy,
    FillMissingValuesStrategy
)
from preprocessing.duplicate_handler import (
    DuplicateHandler,
    DropAllDuplicatesStrategy,
    KeepFirstDuplicateStrategy,
    KeepLastDuplicateStrategy
)
from preprocessing.outlier_detection import (
    OutlierDetector,
    ZScoreOutlierDetection,
    IQROutlierDetection
)
from preprocessing.feature_engineering import (
    FeatureEngineer,
    OneHotEncoding,
    LogTransformation,
    StandardScaling,
    MinMaxScaling
)

# Import analysis modules
from analysis.analysis_setup.basic_data_inspection import (
    DataInspector,
    DataTypesInspectionStrategy,
    SummaryStatisticsInspectionStrategy
)
from analysis.analysis_setup.missing_values_analysis import SimpleMissingValuesAnalysis
from analysis.analysis_setup.univariate_analysis import (
    UnivariateAnalyzer,
    NumericalUnivariateAnalysis,
    CategoricalUnivariateAnalysis
)
from analysis.analysis_setup.bivariate_analysis import (
    BivariateAnalyzer,
    NumericalVsNumericalAnalysis,
    CategoricalVsNumericalAnalysis
)
from analysis.analysis_setup.multivariate_analysis import SimpleMultivariateAnalysis

# Page configuration
st.set_page_config(
    page_title="Data Preprocessing App",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_current' not in st.session_state:
    st.session_state.df_current = None
if 'steps_completed' not in st.session_state:
    st.session_state.steps_completed = {
        'upload': False,
        'analysis': False,
        'cleaning': False,
        'outliers': False,
        'encoding': False,
        'feature_engineering': False,
        'scaling': False
    }

# Sidebar navigation
st.sidebar.title("🔧 Data Preprocessing Pipeline")
st.sidebar.markdown("---")

# Step selection
steps = [
    "1. Upload Data",
    "2. Data Analysis",
    "3. Data Cleaning",
    "4. Outlier Detection",
    "5. Categorical Encoding",
    "6. Feature Engineering",
    "7. Feature Scaling",
    "8. Download Data"
]

selected_step = st.sidebar.radio("Select Step", steps)

# Progress indicator
st.sidebar.markdown("---")
st.sidebar.subheader("Progress")
progress_steps = [
    ("Upload", st.session_state.steps_completed['upload']),
    ("Analysis", st.session_state.steps_completed['analysis']),
    ("Cleaning", st.session_state.steps_completed['cleaning']),
    ("Outliers", st.session_state.steps_completed['outliers']),
    ("Encoding", st.session_state.steps_completed['encoding']),
    ("Feature Eng", st.session_state.steps_completed['feature_engineering']),
    ("Scaling", st.session_state.steps_completed['scaling']),
]
for step_name, completed in progress_steps:
    status = "✅" if completed else "⏳"
    st.sidebar.write(f"{status} {step_name}")

# Reset button
if st.sidebar.button("🔄 Reset All Data"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.steps_completed = {
        'upload': False,
        'analysis': False,
        'cleaning': False,
        'outliers': False,
        'encoding': False,
        'feature_engineering': False,
        'scaling': False
    }
    st.rerun()

# Main content area
st.title("🔧 Data Preprocessing Application")
st.markdown("Upload your CSV file and preprocess it step by step using the sidebar navigation.")

# Helper function to check if data is loaded
def check_data_loaded():
    if st.session_state.df_current is None:
        st.warning("⚠️ Please upload data first in Step 1: Upload Data")
        return False
    return True

# Helper function to display data info
def display_data_info(df, title="Current Data Info"):
    st.subheader(title)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.markdown("---")

# Step 1: Upload Data
if selected_step == "1. Upload Data":
    st.header("📤 Step 1: Upload Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df_original = df.copy()
            st.session_state.df_current = df.copy()
            st.session_state.steps_completed['upload'] = True
            
            st.success("✅ Data uploaded successfully!")
            display_data_info(df, "Uploaded Data Info")
            
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.subheader("Data Types")
            st.dataframe(pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            }), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        st.info("👆 Please upload a CSV file to get started")

# Step 2: Data Analysis
elif selected_step == "2. Data Analysis":
    st.header("📊 Step 2: Data Analysis")
    
    if not check_data_loaded():
        st.stop()
    
    df = st.session_state.df_current.copy()
    display_data_info(df)
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Basic Inspection", "Missing Values Analysis", "Univariate Analysis", 
         "Bivariate Analysis", "Multivariate Analysis"]
    )
    
    if analysis_type == "Basic Inspection":
        st.subheader("Basic Data Inspection")
        
        tab1, tab2 = st.tabs(["Data Types", "Summary Statistics"])
        
        with tab1:
            st.write("**Data Types and Non-null Counts:**")
            info_str = StringIO()
            df.info(buf=info_str)
            st.text(info_str.getvalue())
        
        with tab2:
            st.write("**Summary Statistics (Numerical Features):**")
            st.dataframe(df.describe(), use_container_width=True)
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                st.write("**Summary Statistics (Categorical Features):**")
                st.dataframe(df[categorical_cols].describe(), use_container_width=True)
    
    elif analysis_type == "Missing Values Analysis":
        st.subheader("Missing Values Analysis")
        
        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing_count.index,
            'Missing Count': missing_count.values,
            'Missing Percentage': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=missing_df, x='Column', y='Missing Count', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_title("Missing Values by Column")
            st.pyplot(fig)
            plt.close()
        else:
            st.success("✅ No missing values found in the dataset!")
    
    elif analysis_type == "Univariate Analysis":
        st.subheader("Univariate Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(numeric_cols) > 0:
            selected_numeric = st.selectbox("Select Numerical Feature", numeric_cols)
            if selected_numeric:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df[selected_numeric], kde=True, bins=30, ax=ax)
                ax.set_title(f"Distribution of {selected_numeric}")
                ax.set_xlabel(selected_numeric)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
                plt.close()
        
        if len(categorical_cols) > 0:
            selected_categorical = st.selectbox("Select Categorical Feature", categorical_cols)
            if selected_categorical:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(x=selected_categorical, data=df, ax=ax, palette="muted")
                ax.set_title(f"Distribution of {selected_categorical}")
                ax.set_xlabel(selected_categorical)
                ax.set_ylabel("Count")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
                plt.close()
    
    elif analysis_type == "Bivariate Analysis":
        st.subheader("Bivariate Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        analysis_subtype = st.radio(
            "Analysis Type",
            ["Numerical vs Numerical", "Categorical vs Numerical"],
            horizontal=True
        )
        
        if analysis_subtype == "Numerical vs Numerical":
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    feature1 = st.selectbox("Select Feature 1", numeric_cols)
                with col2:
                    feature2 = st.selectbox("Select Feature 2", numeric_cols, 
                                           index=min(1, len(numeric_cols)-1))
                
                if feature1 and feature2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x=feature1, y=feature2, data=df, ax=ax)
                    ax.set_title(f"{feature1} vs {feature2}")
                    ax.set_xlabel(feature1)
                    ax.set_ylabel(feature2)
                    st.pyplot(fig)
                    plt.close()
        else:
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    cat_feature = st.selectbox("Select Categorical Feature", categorical_cols)
                with col2:
                    num_feature = st.selectbox("Select Numerical Feature", numeric_cols)
                
                if cat_feature and num_feature:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(x=cat_feature, y=num_feature, data=df, ax=ax)
                    ax.set_title(f"{cat_feature} vs {num_feature}")
                    ax.set_xlabel(cat_feature)
                    ax.set_ylabel(num_feature)
                    ax.tick_params(axis='x', rotation=45)
                    st.pyplot(fig)
                    plt.close()
    
    elif analysis_type == "Multivariate Analysis":
        st.subheader("Multivariate Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            selected_cols = st.multiselect(
                "Select Numerical Features for Correlation",
                numeric_cols,
                default=numeric_cols[:min(10, len(numeric_cols))]
            )
            
            if len(selected_cols) > 1:
                corr_matrix = df[selected_cols].corr()
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                           linewidths=0.5, ax=ax, square=True)
                ax.set_title("Correlation Heatmap")
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Please select at least 2 numerical features")
    
    st.session_state.steps_completed['analysis'] = True

# Step 3: Data Cleaning
elif selected_step == "3. Data Cleaning":
    st.header("🧹 Step 3: Data Cleaning")
    
    if not check_data_loaded():
        st.stop()
    
    df = st.session_state.df_current.copy()
    display_data_info(df, "Before Cleaning")
    
    tab1, tab2 = st.tabs(["Missing Values", "Duplicate Rows"])
    
    with tab1:
        st.subheader("Missing Values Handling")
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            st.write(f"**Total missing values:** {missing_count}")
            
            missing_cols = df.columns[df.isnull().any()].tolist()
            st.write(f"**Columns with missing values:** {', '.join(missing_cols)}")
            
            handling_method = st.radio(
                "Select Handling Method",
                ["Drop Missing Values", "Fill Missing Values"],
                horizontal=True
            )
            
            if handling_method == "Drop Missing Values":
                drop_option = st.selectbox(
                    "Drop Option",
                    ["Drop rows with any missing values", 
                     "Drop rows with all missing values",
                     "Drop rows with missing values above threshold"]
                )
                
                if drop_option == "Drop rows with any missing values":
                    threshold = None
                    axis = 0
                elif drop_option == "Drop rows with all missing values":
                    threshold = df.shape[1]
                    axis = 0
                else:
                    threshold = st.number_input(
                        "Minimum number of non-null values required",
                        min_value=1,
                        max_value=df.shape[1],
                        value=df.shape[1] - 1
                    )
                    axis = 0
                
                if st.button("Apply Drop Missing Values"):
                    strategy = DropMissingValuesStrategy(axis=axis, thresh=threshold)
                    handler = MissingValueHandler(strategy)
                    df_cleaned = handler.handle_missing_values(df)
                    st.session_state.df_current = df_cleaned
                    st.success(f"✅ Removed rows. New shape: {df_cleaned.shape}")
                    st.rerun()
            
            else:  # Fill Missing Values
                fill_method = st.selectbox(
                    "Fill Method",
                    ["Mean", "Median", "Mode", "Constant Value"]
                )
                
                fill_value = None
                if fill_method == "Constant Value":
                    fill_value = st.text_input("Enter constant value to fill")
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
                
                if st.button("Apply Fill Missing Values"):
                    strategy = FillMissingValuesStrategy(
                        method=method_map[fill_method],
                        fill_value=fill_value
                    )
                    handler = MissingValueHandler(strategy)
                    df_cleaned = handler.handle_missing_values(df)
                    st.session_state.df_current = df_cleaned
                    st.success("✅ Missing values filled successfully!")
                    st.rerun()
        else:
            st.success("✅ No missing values found in the dataset!")
    
    with tab2:
        st.subheader("Duplicate Rows Handling")
        
        duplicate_handler = DuplicateHandler(DropAllDuplicatesStrategy())
        duplicate_count = duplicate_handler.count_duplicates(df)
        
        if duplicate_count > 0:
            st.write(f"**Number of duplicate rows:** {duplicate_count}")
            
            duplicate_method = st.radio(
                "Select Handling Method",
                ["Remove All Duplicates", "Keep First", "Keep Last"],
                horizontal=True
            )
            
            subset_cols = st.multiselect(
                "Select columns to check for duplicates (leave empty for all columns)",
                df.columns.tolist()
            )
            
            subset = subset_cols if subset_cols else None
            
            if st.button("Apply Duplicate Removal"):
                if duplicate_method == "Remove All Duplicates":
                    strategy = DropAllDuplicatesStrategy()
                elif duplicate_method == "Keep First":
                    strategy = KeepFirstDuplicateStrategy(subset=subset)
                else:
                    strategy = KeepLastDuplicateStrategy(subset=subset)
                
                duplicate_handler.set_strategy(strategy)
                df_cleaned = duplicate_handler.handle_duplicates(df)
                st.session_state.df_current = df_cleaned
                st.success(f"✅ Removed duplicates. New shape: {df_cleaned.shape}")
                st.rerun()
        else:
            st.success("✅ No duplicate rows found in the dataset!")
    
    display_data_info(st.session_state.df_current, "After Cleaning")
    st.session_state.steps_completed['cleaning'] = True

# Step 4: Outlier Detection
elif selected_step == "4. Outlier Detection":
    st.header("🔍 Step 4: Outlier Detection & Treatment")
    
    if not check_data_loaded():
        st.stop()
    
    df = st.session_state.df_current.copy()
    display_data_info(df)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.warning("⚠️ No numerical columns found for outlier detection")
        st.stop()
    
    selected_cols = st.multiselect(
        "Select columns for outlier detection",
        numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))]
    )
    
    if len(selected_cols) > 0:
        detection_method = st.radio(
            "Detection Method",
            ["Z-Score", "IQR (Interquartile Range)"],
            horizontal=True
        )
        
        if detection_method == "Z-Score":
            threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.5)
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
            horizontal=True
        )
        
        if st.button("Apply Outlier Treatment"):
            if treatment_method == "Remove Outliers":
                df_numeric_cleaned = detector.handle_outliers(df_numeric, method="remove")
                df_categorical = df.drop(columns=selected_cols)
                df_categorical_cleaned = df_categorical.loc[df_numeric_cleaned.index]
                df_cleaned = pd.concat([df_numeric_cleaned, df_categorical_cleaned], axis=1)
            else:
                df_numeric_cleaned = detector.handle_outliers(df_numeric, method="cap")
                df_categorical = df.drop(columns=selected_cols)
                df_cleaned = pd.concat([df_numeric_cleaned, df_categorical], axis=1)
            
            # Reorder columns to match original
            df_cleaned = df_cleaned[df.columns]
            st.session_state.df_current = df_cleaned
            st.success(f"✅ Outliers treated. New shape: {df_cleaned.shape}")
            st.rerun()
    
    display_data_info(st.session_state.df_current, "After Outlier Treatment")
    st.session_state.steps_completed['outliers'] = True

# Step 5: Categorical Encoding
elif selected_step == "5. Categorical Encoding":
    st.header("🔤 Step 5: Categorical Encoding")
    
    if not check_data_loaded():
        st.stop()
    
    df = st.session_state.df_current.copy()
    display_data_info(df)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(categorical_cols) == 0:
        st.info("ℹ️ No categorical columns found. Skipping this step.")
        st.session_state.steps_completed['encoding'] = True
    else:
        st.write("**Available categorical columns:**")
        st.write(", ".join(categorical_cols))
        
        selected_cols = st.multiselect(
            "Select categorical columns to encode",
            categorical_cols
        )
        
        if len(selected_cols) > 0:
            st.write(f"**Selected columns:** {', '.join(selected_cols)}")
            
            # Preview before encoding
            st.subheader("Preview Before Encoding")
            st.dataframe(df[selected_cols].head(), use_container_width=True)
            
            if st.button("Apply One-Hot Encoding"):
                try:
                    encoder = FeatureEngineer(OneHotEncoding(features=selected_cols))
                    df_encoded = encoder.apply_feature_engineering(df)
                    st.session_state.df_current = df_encoded
                    st.success("✅ Categorical encoding applied successfully!")
                    
                    st.subheader("Preview After Encoding")
                    # Show some of the new encoded columns
                    encoded_cols = [col for col in df_encoded.columns if any(sel_col in col for sel_col in selected_cols)]
                    st.dataframe(df_encoded[encoded_cols[:min(10, len(encoded_cols))]].head(), use_container_width=True)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during encoding: {str(e)}")
    
    display_data_info(st.session_state.df_current, "After Encoding")
    st.session_state.steps_completed['encoding'] = True

# Step 6: Feature Engineering
elif selected_step == "6. Feature Engineering":
    st.header("⚙️ Step 6: Feature Engineering")
    
    if not check_data_loaded():
        st.stop()
    
    df = st.session_state.df_current.copy()
    display_data_info(df)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.warning("⚠️ No numerical columns found for feature engineering")
        st.stop()
    
    transformation_type = st.selectbox(
        "Select Transformation",
        ["Log Transformation"]
    )
    
    if transformation_type == "Log Transformation":
        st.write("**Log transformation applies log1p(x) = log(1+x) to handle zero values**")
        
        selected_cols = st.multiselect(
            "Select numerical features for log transformation",
            numeric_cols
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
            
            if st.button("Apply Log Transformation"):
                try:
                    transformer = FeatureEngineer(LogTransformation(features=selected_cols))
                    df_transformed = transformer.apply_feature_engineering(df)
                    st.session_state.df_current = df_transformed
                    st.success("✅ Log transformation applied successfully!")
                    
                    st.subheader("Preview After Transformation")
                    st.dataframe(df_transformed[selected_cols].head(), use_container_width=True)
                    st.write("**Statistics:**")
                    st.dataframe(df_transformed[selected_cols].describe(), use_container_width=True)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during transformation: {str(e)}")
    
    display_data_info(st.session_state.df_current, "After Feature Engineering")
    st.session_state.steps_completed['feature_engineering'] = True

# Step 7: Feature Scaling
elif selected_step == "7. Feature Scaling":
    st.header("📏 Step 7: Feature Scaling")
    
    if not check_data_loaded():
        st.stop()
    
    df = st.session_state.df_current.copy()
    display_data_info(df)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.warning("⚠️ No numerical columns found for scaling")
        st.stop()
    
    scaling_method = st.radio(
        "Scaling Method",
        ["Standard Scaling (Z-score)", "Min-Max Scaling"],
        horizontal=True
    )
    
    selected_cols = st.multiselect(
        "Select numerical features to scale",
        numeric_cols
    )
    
    if len(selected_cols) > 0:
        st.subheader("Preview Before Scaling")
        st.dataframe(df[selected_cols].head(), use_container_width=True)
        st.write("**Statistics:**")
        st.dataframe(df[selected_cols].describe(), use_container_width=True)
        
        if scaling_method == "Min-Max Scaling":
            feature_range = st.slider(
                "Feature Range",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.1
            )
        
        if st.button("Apply Scaling"):
            try:
                if scaling_method == "Standard Scaling (Z-score)":
                    scaler = FeatureEngineer(StandardScaling(features=selected_cols))
                else:
                    scaler = FeatureEngineer(MinMaxScaling(features=selected_cols, feature_range=feature_range))
                
                df_scaled = scaler.apply_feature_engineering(df)
                st.session_state.df_current = df_scaled
                st.success("✅ Feature scaling applied successfully!")
                
                st.subheader("Preview After Scaling")
                st.dataframe(df_scaled[selected_cols].head(), use_container_width=True)
                st.write("**Statistics:**")
                st.dataframe(df_scaled[selected_cols].describe(), use_container_width=True)
                st.rerun()
            except Exception as e:
                st.error(f"Error during scaling: {str(e)}")
    
    display_data_info(st.session_state.df_current, "After Scaling")
    st.session_state.steps_completed['scaling'] = True

# Step 8: Download Data
elif selected_step == "8. Download Data":
    st.header("💾 Step 8: Download Preprocessed Data")
    
    if not check_data_loaded():
        st.stop()
    
    df_final = st.session_state.df_current.copy()
    
    st.subheader("Final Data Summary")
    display_data_info(df_final, "Final Preprocessed Data")
    
    st.subheader("Data Preview")
    st.dataframe(df_final.head(20), use_container_width=True)
    
    st.subheader("Download Preprocessed Data")
    
    # Convert dataframe to CSV
    csv = df_final.to_csv(index=False)
    
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="preprocessed_data.csv",
        mime="text/csv"
    )
    
    st.success("✅ Your data is ready for download!")

# Footer
st.markdown("---")
st.markdown("**Data Preprocessing Application** - Process your data step by step without writing code!")

