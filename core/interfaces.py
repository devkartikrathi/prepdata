from abc import ABC, abstractmethod
import pandas as pd
import streamlit as st

class BaseStep(ABC):
    """Base interface for all pipeline steps."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The display name of the step."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A brief description of what this step does."""
        pass
    
    @property
    def group(self) -> str:
        """The group this step belongs to (e.g., 'Preprocessing', 'Analysis')."""
        return "General"

class PreprocessingStep(BaseStep):
    """Interface for preprocessing steps that modify the data."""
    
    @property
    def group(self) -> str:
        return "Preprocessing"

    @abstractmethod
    def render_ui(self, df: pd.DataFrame) -> dict:
        """
        Renders the configuration UI for this step.
        Returns a dictionary of parameters to be passed to execute.
        """
        pass

    @abstractmethod
    def execute(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        Executes the preprocessing step.
        Returns the modified DataFrame.
        """
        pass

class AnalysisStep(BaseStep):
    """Interface for analysis steps that inspect the data."""
    
    @property
    def group(self) -> str:
        return "Analysis"

    @abstractmethod
    def render(self, df: pd.DataFrame):
        """
        Renders the analysis results directly to the Streamlit app.
        """
        pass
