import os
import json
import pandas as pd
import numpy as np
import base64
from io import BytesIO, StringIO
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.messages import HumanMessage
import pickle
import inspect
from flask import Flask, request, jsonify, send_file, render_template

# Configuration
UPLOAD_FOLDER = 'uploads'
METADATA_FOLDER = 'metadata'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
DEFAULT_DATA_FILE = 'supplychain_data.csv'  # Default data file
google_api_key = os.getenv("GOOGLE_API_KEY")  # API key from environment variable   

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(METADATA_FOLDER, exist_ok=True)

LATEST_FILE_PATH = None
LATEST_METADATA = None

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_metadata(df):
    """Extract metadata from a dataframe with a simpler approach"""
    # Basic dataset information
    metadata = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "sample_data": df.head(3).to_dict(orient='records')
    }
    
    # Simple data type categorization
    metadata["column_types"] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            metadata["column_types"][col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            metadata["column_types"][col] = "datetime"
        else:
            metadata["column_types"][col] = "categorical"
    
    # Basic statistics for numeric columns only
    metadata["numeric_stats"] = {}
    numeric_cols = [col for col in df.columns if metadata["column_types"][col] == "numeric"]
    if numeric_cols:
        stats_df = df[numeric_cols].describe().transpose()
        for col in numeric_cols:
            if col in stats_df.index:
                metadata["numeric_stats"][col] = {
                    "min": float(stats_df.loc[col, "min"]),
                    "max": float(stats_df.loc[col, "max"]),
                    "mean": float(stats_df.loc[col, "mean"]),
                    "std": float(stats_df.loc[col, "std"])
                }
    
    # For categorical columns, get unique values count
    metadata["categorical_stats"] = {}
    cat_cols = [col for col in df.columns if metadata["column_types"][col] == "categorical"]
    for col in cat_cols:
        unique_values = df[col].nunique()
        metadata["categorical_stats"][col] = {
            "unique_values": min(unique_values, 100),  # Limit to 100 for large categories
            "top_categories": df[col].value_counts().head(5).to_dict() if unique_values < 100 else {}
        }
    
    return metadata

def save_metadata(df):
    """Extract metadata and save it to memory"""
    global LATEST_METADATA
    LATEST_METADATA = extract_metadata(df)
    return LATEST_METADATA

def get_latest_metadata():
    """Get the latest metadata from memory"""
    global LATEST_METADATA
    return LATEST_METADATA

def load_dataframe():
    """Load the most recent dataframe for code execution or default if none exists"""
    global LATEST_FILE_PATH
    
    if not LATEST_FILE_PATH:
        # Check for uploaded files
        upload_files = [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)]
        
        if not upload_files:
            # No uploads found, use default data file
            if os.path.exists(DEFAULT_DATA_FILE):
                LATEST_FILE_PATH = DEFAULT_DATA_FILE
                print(f"Using default data file: {DEFAULT_DATA_FILE}")
            else:
                return None
        else:
            # Use most recent upload
            latest_file = max(upload_files, key=lambda f: os.path.getctime(os.path.join(UPLOAD_FOLDER, f)))
            LATEST_FILE_PATH = os.path.join(UPLOAD_FOLDER, latest_file)
    
    if LATEST_FILE_PATH.endswith('.csv'):
        return pd.read_csv(LATEST_FILE_PATH)
    elif LATEST_FILE_PATH.endswith(('.xlsx', '.xls')):
        return pd.read_excel(LATEST_FILE_PATH)
    else:
        return None

def generate_analysis_code(query, metadata):
    """Generate executable code for analysis and visualization based on the query"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
    
    # Create a prompt that instructs the model to generate appropriate analysis code
    prompt = f"""
    You are an expert data analyst for supply chain analytics. Generate Python code to answer the following question.
    
    IMPORTANT: Use ONLY the existing 'df' variable that already contains the loaded dataset. DO NOT create sample data or new dataframes except for transformations of the existing df.
    
    Dataset information:
    - Columns: {', '.join(metadata['columns'])}
    - Data types: {json.dumps(metadata['column_types'])}
    - Number of rows: {metadata['row_count']}
    
    Sample data (first 3 rows):
    {json.dumps(metadata['sample_data'], indent=2)}
    
    User question: {query}
    
    Based on the query and data, determine which type of response would be most appropriate:
    1. A chart or graph (using matplotlib/seaborn)
    2. A table of data (using pandas)
    3. A text summary/analysis
    4. A combination of the above
    
    Generate valid, executable Python code that:
    1. Processes the EXISTING 'df' dataframe that is already loaded
    2. Creates appropriate visualizations if needed (save to 'plt' object)
    3. Returns results in the most informative format
    
    For visualizations:
    - Use appropriate chart types (bar, line, pie, etc.)
    - Include proper titles, labels, and legends
    - Format the chart to be readable and professional
    - Set plt.tight_layout() for proper spacing
    
    Return ONLY the Python code with no explanations or markdown. The code must:
    - Handle potential errors like missing data
    - Not use external libraries beyond pandas, numpy, matplotlib, and seaborn
    - Not create new sample data - use only the existing 'df' variable
    - End with a visualization or a return statement for data to be displayed
    - Include any analysis text as comments for later extraction
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    code = response.content
    
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].strip()
    
    return code

def encode_plot_to_base64():
    """Convert matplotlib plot to base64 for displaying in web/apps"""
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

def get_plot_as_image():
    """Return matplotlib plot as an image in bytes"""
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    return buf

def execute_analysis_code(code):
    """Safely execute the generated analysis code and capture outputs"""
    df = load_dataframe()
    if df is None:
        return {"error": "Cannot load the dataset"}
    print(f"code generate {code}")
    restricted_globals = {
        'pd': pd,
        'df': df,
        'np': np,
        'plt': plt,
        'sns': sns,
        'print': print
    }
    
    try:
        # Capture printed output
        old_stdout = sys.stdout
        mystdout = StringIO()
        sys.stdout = mystdout
        
        # Execute code
        local_vars = {}
        exec(code, restricted_globals, local_vars)
        
        sys.stdout = old_stdout
        printed_output = mystdout.getvalue()
        
        response = {
            "code_executed": code,
            "printed_output": printed_output if printed_output else None
        }
        
        # Check if there's a matplotlib figure to encode
        if plt.get_fignums():
            response["visualization"] = encode_plot_to_base64()
            response["has_visualization"] = True
            response["plot_bytes"] = get_plot_as_image()
        else:
            response["has_visualization"] = False
        
        # Check for data results
        result = None
        for var_name, var_val in local_vars.items():
            if not var_name.startswith('__'):
                result = var_val
                break
        
        if result is not None:
            if isinstance(result, pd.DataFrame):
                if len(result) > 20:
                    response["data"] = result.head(20).to_dict(orient='records')
                    response["note"] = f"Showing first 20 rows of {len(result)} total rows"
                else:
                    response["data"] = result.to_dict(orient='records')
            elif isinstance(result, pd.Series):
                response["data"] = result.to_dict()
            else:
                response["data"] = result
            
        return response
        
    except Exception as e:
        plt.close('all')  # Close any open figures in case of error
        return {"error": str(e), "code_attempted": code}

def extract_insights_from_analysis(analysis_results, query):
    """Extract key insights from the analysis results using LLM"""
    if "error" in analysis_results:
        return analysis_results
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
    
    # Create data context for the LLM
    context = {}
    if analysis_results.get("data") is not None:
        context["data"] = analysis_results["data"]
    if analysis_results.get("printed_output") is not None:
        context["printed_output"] = analysis_results["printed_output"]
    
    has_viz = analysis_results.get("has_visualization", False)
    
    prompt = f"""
    User question: {query}
    
    Analysis results: {json.dumps(context)}
    
    Based on the analysis results, provide a concise summary of the key insights.
    Focus on directly answering the user's question in a business-friendly way.
    
    {"Note: This analysis includes a visualization that is not shown here." if has_viz else ""}
    
    Keep your response under 150 words and highlight the most important findings first.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Add the insights to the results
    analysis_results["insights"] = response.content
    
    return analysis_results

def ask_query(query):
    """Query the data using generated code for analysis and visualization"""
    try:
        # Load metadata for context
        metadata = get_latest_metadata()
        
        # If no metadata exists, try to load the default file
        if not metadata:
            df = load_dataframe()
            if df is not None:
                metadata = save_metadata(df)
            else:
                return {"error": "No data available. Please upload a file or place a default file named 'supplychain_data.csv' in the root directory."}
        
        # Generate and execute code for analysis/visualization
        code = generate_analysis_code(query, metadata)
        results = execute_analysis_code(code)
        
        # Extract key insights using LLM
        if "error" not in results:
            results = extract_insights_from_analysis(results, query)
        
        return results
            
    except Exception as e:
        return {"error": str(e)}

def process_file(file_path):
    """Process a new file and extract its metadata"""
    global LATEST_FILE_PATH
    
    LATEST_FILE_PATH = file_path
    
    # Load the dataframe
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        return {"error": "Unsupported file format"}
    
    # Extract and save metadata
    metadata = save_metadata(df)
    
    return {
        "status": "success",
        "rows": len(df),
        "columns": len(df.columns)
    }

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        result = process_file(file_path)
        return jsonify(result)
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    
    query_text = data['query']
    result = ask_query(query_text)
    
    return jsonify(result)

@app.route('/query_image', methods=['POST'])
def query_image():
    """Endpoint to get visualization as an image from a query"""
    data = request.json
    process_file("supplychain_data.csv")  # Ensure metadata is loaded before querying
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    
    query_text = data['query']
    result = ask_query(query_text)
    
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    
    if not result.get("has_visualization", False):
        return jsonify({"error": "No visualization generated for this query"}), 400
    
    # Return the image directly
    return send_file(
        result["plot_bytes"],
        mimetype='image/png'
    )

@app.route('/query_with_insights', methods=['POST'])
def query_with_insights():
    """Endpoint to get both image and insights from a query"""
    data = request.json
    process_file("supplychain_data.csv")
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    
    query_text = data['query']
    result = ask_query(query_text)
    
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    
    # Return insights and base64 encoded image
    response = {
        "insights": result.get("insights", "No insights available"),
        "has_visualization": result.get("has_visualization", False)
    }
    
    if result.get("has_visualization", False):
        response["visualization_base64"] = result["visualization"]
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, port=5000)