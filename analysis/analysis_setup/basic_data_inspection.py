from abc import ABC, abstractmethod
import pandas as pd
import streamlit as st
from io import StringIO
from core.interfaces import AnalysisStep

class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        pass


class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print("\nData Types and Non-null Counts:")
        print(df.info())


class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["O"]))


class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        self._strategy.inspect(df)


class BasicInspectionStep(AnalysisStep):
    @property
    def name(self) -> str:
        return "Basic Inspection"

    @property
    def description(self) -> str:
        return "View data types, non-null counts, and summary statistics."

    def render(self, df: pd.DataFrame):
        st.subheader("Basic Data Inspection")
        
        if df is None or df.empty:
            st.warning("The DataFrame is empty.")
            return

        tab1, tab2 = st.tabs(["Data Types", "Summary Statistics"])
        
        with tab1:
            st.write("**Data Types and Non-null Counts:**")
            
            # Create a DataFrame for info instead of using text output
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Non-Null Count': df.count().values,
                'Dtype': df.dtypes.values.astype(str)
            })
            st.dataframe(info_df, use_container_width=True)
            
            st.markdown(f"**Total Rows:** {df.shape[0]}")
            st.markdown(f"**Total Columns:** {df.shape[1]}")
            st.markdown(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        with tab2:
            st.write("**Summary Statistics (Numerical Features):**")
            describe_num = df.describe()
            if not describe_num.empty:
                st.dataframe(describe_num, use_container_width=True)
            else:
                st.info("No numerical features found.")
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                st.write("**Summary Statistics (Categorical Features):**")
                st.dataframe(df[categorical_cols].describe(), use_container_width=True)
            else:
                st.info("No categorical features found.")
