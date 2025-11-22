import streamlit as st
import pandas as pd
import sys
import os
import io
from pathlib import Path
from google import genai

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.registry import PluginRegistry
from core.interfaces import PreprocessingStep, AnalysisStep

# Page configuration
st.set_page_config(
    page_title="Prep Data",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize registry
@st.cache_resource
def get_registry():
    registry = PluginRegistry()
    registry.discover_plugins(str(Path(__file__).parent))
    return registry

registry = get_registry()

# Define the strict order of steps
# We use the 'name' property of the steps to identify them
PIPELINE_ORDER = [
    "Upload Data",
    "Basic Inspection",
    "Missing Values Handling",
    "Duplicate Rows Handling",
    "Outlier Detection",
    "Categorical Encoding",
    "Feature Engineering",
    "Feature Scaling",
    "Download Data"
]

# Initialize session state
if 'current_step_index' not in st.session_state:
    st.session_state.current_step_index = 0

# Checkpoints: Dictionary mapping step index to the DataFrame state at the BEGINNING of that step
# step 0 (Upload) -> No DF yet
# step 1 (Inspection) -> DF loaded
if 'checkpoints' not in st.session_state:
    st.session_state.checkpoints = {}

if 'df_current' not in st.session_state:
    st.session_state.df_current = None

def save_checkpoint(step_index, df):
    """Save the dataframe state for a specific step index."""
    st.session_state.checkpoints[step_index] = df.copy()

def get_checkpoint(step_index):
    """Get the dataframe state for a specific step index."""
    return st.session_state.checkpoints.get(step_index)

def revert_to_step(step_index):
    """Revert the current state to the beginning of the specified step."""
    if step_index in st.session_state.checkpoints:
        st.session_state.df_current = st.session_state.checkpoints[step_index].copy()
        # Clear future checkpoints
        keys_to_remove = [k for k in st.session_state.checkpoints if k > step_index]
        for k in keys_to_remove:
            del st.session_state.checkpoints[k]
        st.success(f"Reverted to state at {PIPELINE_ORDER[step_index]}")
        st.rerun()

# Helper function to find step class by name
def get_step_class(name):
    all_steps = registry.get_steps("Preprocessing") + registry.get_steps("Analysis")
    for step_cls in all_steps:
        if step_cls().name == name:
            return step_cls
    return None

# Sidebar Progress
st.sidebar.title("Prep Data")
st.sidebar.markdown("---")

# Progress Bar
progress = st.session_state.current_step_index / (len(PIPELINE_ORDER) - 1)
st.sidebar.progress(progress)

st.sidebar.subheader("Steps")
for i, step_name in enumerate(PIPELINE_ORDER):
    if i < st.session_state.current_step_index:
        icon = "✅"
    elif i == st.session_state.current_step_index:
        icon = "👉"
    else:
        icon = "⏳"
    
    # Make steps clickable if we have visited them (checkpoint exists)
    if i in st.session_state.checkpoints or i == 0:
        if st.sidebar.button(f"{icon} {step_name}", key=f"nav_{i}"):
            st.session_state.current_step_index = i
            if i in st.session_state.checkpoints:
                 st.session_state.df_current = st.session_state.checkpoints[i].copy()
            st.rerun()
    else:
        st.sidebar.write(f"{icon} {step_name}")

# Main Content
current_step_name = PIPELINE_ORDER[st.session_state.current_step_index]
st.title(f"{current_step_name}")

# AI Assistant Setup
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.environ.get("GOOGLE_API_KEY", "")

def get_ai_suggestion(step_name, df_info, step_description):
    if not st.session_state.gemini_api_key:
        return "Please provide a Google Gemini API Key to use the AI Assistant."
    
    try:
        client = genai.Client(api_key=st.session_state.gemini_api_key)
        
        prompt = f"""
        You are a helpful data science assistant. The user is using a data preprocessing tool.
        
        Current Step: {step_name}
        Step Description: {step_description}
        
        Data Summary:
        {df_info}
        
        The user needs suggestions on what to do in this step given the current data. 
        Provide concise, actionable advice. If there are issues (like missing values or outliers), highlight them.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error communicating with AI: {e}"

# AI Assistant UI
with st.expander("🤖 Ask AI Assistant for Help"):
    if not st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = st.text_input("Enter Google Gemini API Key", type="password")
    
    if st.button("✨ Get Suggestions"):
        with st.spinner("Thinking..."):
            # Gather context
            df_context = "No data loaded yet."
            if st.session_state.df_current is not None:
                buffer = io.StringIO()
                st.session_state.df_current.info(buf=buffer)
                info_str = buffer.getvalue()
                head_str = st.session_state.df_current.head().to_string()
                
                # Add missing value counts
                missing_counts = st.session_state.df_current.isnull().sum().to_string()
                
                df_context = f"Info:\n{info_str}\n\nMissing Values:\n{missing_counts}\n\nHead:\n{head_str}"
            
            # Get description if available
            step_desc = "Perform this step."
            step_cls = get_step_class(current_step_name)
            if step_cls:
                step_desc = step_cls().description
            
            suggestion = get_ai_suggestion(current_step_name, df_context, step_desc)
            st.markdown(suggestion)

# Navigation Buttons Helper
def nav_buttons(can_go_next=True):
    col1, col2, col3 = st.columns([1, 5, 1])
    
    with col1:
        if st.session_state.current_step_index > 0:
            if st.button("⬅️ Previous"):
                st.session_state.current_step_index -= 1
                # Restore state of previous step
                prev_idx = st.session_state.current_step_index
                if prev_idx in st.session_state.checkpoints:
                    st.session_state.df_current = st.session_state.checkpoints[prev_idx].copy()
                st.rerun()
    
    with col3:
        if st.session_state.current_step_index < len(PIPELINE_ORDER) - 1:
            if st.button("Next ➡️", disabled=not can_go_next):
                # Save checkpoint for the NEXT step (which starts with current DF)
                next_idx = st.session_state.current_step_index + 1
                save_checkpoint(next_idx, st.session_state.df_current)
                st.session_state.current_step_index += 1
                st.rerun()

# Step Logic
if current_step_name == "Upload Data":
    st.markdown("Start by uploading your CSV file.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            if st.session_state.df_current is None:
                df = pd.read_csv(uploaded_file)
                st.session_state.df_current = df
                save_checkpoint(0, df) # Save initial state
                save_checkpoint(1, df) # Ready for next step
            
            st.success("File uploaded successfully!")
            st.dataframe(st.session_state.df_current.head())
            nav_buttons(can_go_next=True)
            
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        nav_buttons(can_go_next=False)

elif current_step_name == "Download Data":
    st.markdown("Your data is ready!")
    if st.session_state.df_current is not None:
        st.dataframe(st.session_state.df_current.head())
        
        csv = st.session_state.df_current.to_csv(index=False)
        st.download_button(
            label="📥 Download Processed CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )
    nav_buttons(can_go_next=False)

else:
    # Dynamic Steps (Analysis or Preprocessing)
    if st.session_state.df_current is None:
        st.error("No data found. Please go back to Upload Data.")
        nav_buttons(can_go_next=False)
    else:
        # Revert Option
        if st.button("🔄 Revert Changes in this Step"):
            revert_to_step(st.session_state.current_step_index)

        step_cls = get_step_class(current_step_name)
        
        if step_cls:
            step_instance = step_cls()
            st.markdown(f"_{step_instance.description}_")
            
            if isinstance(step_instance, AnalysisStep):
                step_instance.render(st.session_state.df_current)
                nav_buttons(can_go_next=True)
                
            elif isinstance(step_instance, PreprocessingStep):
                params = step_instance.render_ui(st.session_state.df_current)
                
                if params:
                    if st.button(f"Apply {current_step_name}"):
                        try:
                            new_df = step_instance.execute(st.session_state.df_current, params)
                            st.session_state.df_current = new_df
                            st.success("Applied successfully! Click Next to proceed.")
                            
                            st.markdown("### 🔍 After State Preview")
                            st.dataframe(new_df.head(), use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error: {e}")
                
                nav_buttons(can_go_next=True)
        else:
            st.error(f"Step implementation '{current_step_name}' not found.")
            nav_buttons(can_go_next=True) # Allow skipping broken steps

# Debug info (Optional, can be removed)
# st.sidebar.markdown("---")
# st.sidebar.json(st.session_state.checkpoints.keys())

