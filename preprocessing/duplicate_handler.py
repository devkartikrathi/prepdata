from abc import ABC, abstractmethod
import pandas as pd
import streamlit as st
from core.interfaces import PreprocessingStep


class DuplicateHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class DropAllDuplicatesStrategy(DuplicateHandlingStrategy):
    """Remove all duplicate rows from the dataframe."""
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()


class KeepFirstDuplicateStrategy(DuplicateHandlingStrategy):
    """Keep the first occurrence of duplicate rows."""
    
    def __init__(self, subset=None):
        self.subset = subset
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(subset=self.subset, keep='first')


class KeepLastDuplicateStrategy(DuplicateHandlingStrategy):
    """Keep the last occurrence of duplicate rows."""
    
    def __init__(self, subset=None):
        self.subset = subset
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(subset=self.subset, keep='last')


class DuplicateHandler:
    """Handler class for managing duplicate row removal strategies."""
    
    def __init__(self, strategy: DuplicateHandlingStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: DuplicateHandlingStrategy):
        self._strategy = strategy
    
    def handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._strategy.handle(df)
    
    def count_duplicates(self, df: pd.DataFrame, subset=None) -> int:
        """Count the number of duplicate rows in the dataframe."""
        if subset is None:
            return df.duplicated().sum()
        else:
            return df.duplicated(subset=subset).sum()


if __name__ == "__main__":
    # Example usage
    data = {
        'A': [1, 2, 2, 3, 3, 3],
        'B': [10, 20, 20, 30, 30, 30],
        'C': ['x', 'y', 'y', 'z', 'z', 'z']
    }
    df = pd.DataFrame(data)
    print(f"Original data shape: {df.shape}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    handler = DuplicateHandler(DropAllDuplicatesStrategy())
    df_cleaned = handler.handle_duplicates(df)
    handler = DuplicateHandler(DropAllDuplicatesStrategy())
    df_cleaned = handler.handle_duplicates(df)
    print(f"After removing duplicates: {df_cleaned.shape}")

class DuplicateHandlingStep(PreprocessingStep):
    @property
    def name(self) -> str:
        return "Duplicate Rows Handling"

    @property
    def description(self) -> str:
        return "Identify and remove duplicate rows from the dataset."

    def render_ui(self, df: pd.DataFrame) -> dict:
        st.subheader("Duplicate Rows Handling")
        
        duplicate_handler = DuplicateHandler(DropAllDuplicatesStrategy())
        duplicate_count = duplicate_handler.count_duplicates(df)
        
        if duplicate_count == 0:
            st.success("✅ No duplicate rows found in the dataset!")
            return None
            
        st.write(f"**Number of duplicate rows:** {duplicate_count}")
        
        duplicate_method = st.radio(
            "Select Handling Method",
            ["Remove All Duplicates", "Keep First", "Keep Last"],
            horizontal=True,
            key="dup_method"
        )
        
        subset_cols = st.multiselect(
            "Select columns to check for duplicates (leave empty for all columns)",
            df.columns.tolist(),
            key="dup_subset"
        )
        
        params = {
            "method": duplicate_method,
            "subset": subset_cols if subset_cols else None
        }
        
        return params

    def execute(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        if params is None:
            return df
            
        method = params.get("method")
        subset = params.get("subset")
        
        duplicate_handler = DuplicateHandler(DropAllDuplicatesStrategy())
        
        if method == "Remove All Duplicates":
            strategy = DropAllDuplicatesStrategy()
        elif method == "Keep First":
            strategy = KeepFirstDuplicateStrategy(subset=subset)
        else:
            strategy = KeepLastDuplicateStrategy(subset=subset)
        
        duplicate_handler.set_strategy(strategy)
        return duplicate_handler.handle_duplicates(df)

