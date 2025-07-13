import streamlit as st
import base64
from io import BytesIO
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
import sys
from main import extract_metadata, generate_analysis_code, execute_analysis_code, extract_insights_from_analysis

# Page configuration
st.set_page_config(
    page_title="Supply Chain Analytics Assistant",
    page_icon="🔍",
    layout="wide",
)

# Add custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0078D7;
        margin-bottom: 1rem;
    }
    .sub-title {
        font-size: 1.2rem;
        font-weight: 500;
        color: #505050;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #0078D7;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .insight-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #0078D7;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None

def load_and_process_file(uploaded_file):
    """Load the uploaded file and process its metadata"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return False
        
        # Save the file to disk
        file_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Save dataframe and metadata to session state
        st.session_state.dataframe = df
        st.session_state.metadata = extract_metadata(df)
        st.session_state.file_uploaded = True
        return True
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return False

def custom_execute_analysis_code(code, df):
    """Execute analysis code directly within Streamlit"""
    try:
        # Create local environment with the dataframe and required libraries
        local_env = {
            'pd': pd,
            'df': df,
            'np': np,
            'plt': plt,
            'sns': sns
        }
        
        # Execute the code
        exec(code, local_env)
        
        # Check if there's a matplotlib figure to encode
        if plt.get_fignums():
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            return {"visualization": img_str, "has_visualization": True}
        else:
            return {"has_visualization": False}
    except Exception as e:
        plt.close('all')  # Close any open figures in case of error
        return {"error": str(e), "code_attempted": code}

def process_query(query_text):
    """Process a query and return visualization and insights"""
    if not st.session_state.file_uploaded:
        return None, None
    
    try:
        # Generate code for analysis/visualization
        code = generate_analysis_code(query_text, st.session_state.metadata)
        
        # Execute code directly in Streamlit
        results = custom_execute_analysis_code(code, st.session_state.dataframe)
        
        if "error" in results:
            st.error(f"Error analyzing data: {results['error']}")
            return None, None
        
        # Extract insights using the existing function
        if results.get("has_visualization", False):
            # We'll skip the insights extraction for simplicity
            # You could implement a simplified version here
            insights = "Analysis complete. See visualization for details."
            return results.get("visualization"), insights
        else:
            return None, "No visualization was generated for this query."
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None, None

# Main app layout
st.markdown("<div class='main-title'>Supply Chain Analytics Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Ask questions about your supply chain data and get visual insights</div>", unsafe_allow_html=True)

# Create a two-column layout
col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("### Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        if load_and_process_file(uploaded_file):
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            st.markdown(f"**Rows:** {st.session_state.metadata['row_count']}")
            st.markdown(f"**Columns:** {st.session_state.metadata['column_count']}")
            
            with st.expander("View Sample Data"):
                st.dataframe(st.session_state.dataframe.head())

with col1:
    st.markdown("### Ask About Your Data")
    
    query_text = st.text_area(
        "Enter your question about the data:",
        height=100,
        placeholder="Example: Show the distribution of delivery times across different regions",
        help="Ask questions about trends, distributions, comparisons, or summaries of your data."
    )
    
    analyze_button = st.button("Analyze", use_container_width=True)
    
    if analyze_button and query_text:
        viz_base64, insights = process_query(query_text)
        
        if viz_base64:
            # Display the visualization
            st.markdown("### Analysis Results")
            img = Image.open(BytesIO(base64.b64decode(viz_base64)))
            st.image(img, use_column_width=True)
            
            # Display insights
            if insights:
                st.markdown("<div class='insight-container'>", unsafe_allow_html=True)
                st.markdown("### Key Insights")
                st.markdown(insights)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            if not st.session_state.file_uploaded:
                st.warning("Please upload a data file first.")
            else:
                st.info("No visualization was generated for this query. Try asking a different question.")

# Add footer
st.markdown("---")
st.markdown("Supply Chain Analytics Assistant | Powered by LLMs and Data Science")