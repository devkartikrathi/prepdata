from abc import ABC, abstractmethod
import pandas as pd


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
    print(f"After removing duplicates: {df_cleaned.shape}")

