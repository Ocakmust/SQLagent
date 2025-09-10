import streamlit as st
import pandas as pd
import os
from pathlib import Path
import base64
from typing import Dict, Any, Optional
from langchain_groq import ChatGroq
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

try:
    from router_agent import LangGraphIntelligentRouter
    from oldagents.plot_agent import VisualizationAgent
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

st.set_page_config(
    page_title="Data Analysis & Visualization Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'router_agent': None,
        'viz_agent': None,
        'generated_dataframe': None,
        'raw_dataframe': None,
        'groq_api_key': '',
        'config_loaded': False,
        'last_query': '',
        'plot_paths': [],
        'initialization_complete': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def load_configuration():
    """Load and validate configuration"""
    st.sidebar.header("Configuration")
    
    api_key = st.sidebar.text_input(
        "GROQ API Key", 
        value=st.session_state.groq_api_key,
        type="password",
        help="Enter your GROQ API key"
    )
    
    if api_key:
        st.session_state.groq_api_key = api_key
        os.environ["GROQ_API_KEY"] = api_key
    
    st.sidebar.subheader("Database Settings")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        db_host = st.text_input("Host", value="localhost")
        db_user = st.text_input("User", value="postgres")
        db_port = st.text_input("Port", value="5432")
    
    with col2:
        db_name = st.text_input("Database", value="musteri_db")
        db_password = st.text_input("Password", type="password", value="123")
    
    st.sidebar.subheader("File Uploads")
    
    csv_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])
    doc_file = st.sidebar.file_uploader("Upload Documentation (PDF)", type=['pdf'])
    column_info_file = st.sidebar.file_uploader("Upload Column Info", type=['txt', 'pdf', 'docx'])
    
    config = {
        'db_params': {
            "host": db_host,
            "database": db_name,
            "user": db_user,
            "password": db_password,
            "port": db_port
        },
        'csv_path': None,
        'doc_path': None,
        'column_info': None
    }
    
    if csv_file:
        csv_path = f"temp_{csv_file.name}"
        with open(csv_path, "wb") as f:
            f.write(csv_file.getbuffer())
        config['csv_path'] = csv_path
        try:
            st.session_state.raw_dataframe = pd.read_csv(csv_path)
        except Exception as e:
            st.sidebar.error(f"CSV loading error: {e}")
    
    if doc_file:
        doc_path = f"temp_{doc_file.name}"
        with open(doc_path, "wb") as f:
            f.write(doc_file.getbuffer())
        config['doc_path'] = doc_path
    
    if column_info_file:
        col_info_path = f"temp_{column_info_file.name}"
        with open(col_info_path, "wb") as f:
            f.write(column_info_file.getbuffer())
        config['column_info'] = col_info_path
    
    return config, api_key

def initialize_agents(config, api_key):
    if not api_key:
        st.sidebar.warning("Please provide GROQ API Key")
        return False
    
    if not AGENTS_AVAILABLE:
        st.sidebar.error("Agent modules not available")
        return False
    
    try:
        llm = ChatGroq(
            model_name="openai/gpt-oss-120b",
            api_key=api_key,
            temperature=0.1
        )
        
        with st.sidebar.status("Initializing Router Agent..."):
            st.session_state.router_agent = LangGraphIntelligentRouter(llm=llm, config=config)
        
        if st.session_state.raw_dataframe is not None:
            with st.sidebar.status("Initializing Visualization Agent..."):
                st.session_state.viz_agent = VisualizationAgent(
                    llm=llm,
                    df=st.session_state.raw_dataframe,
                    doc_path=config.get('doc_path'),
                    column_info_path=config.get('column_info')
                )
        
        st.session_state.initialization_complete = True
        st.sidebar.success("Agents initialized successfully!")
        return True
        
    except Exception as e:
        st.sidebar.error(f"Initialization failed: {e}")
        return False

def display_dataframe_tab():
    st.header("DataFrame Query & Results")
    
    if not st.session_state.initialization_complete:
        st.warning("Please complete initialization in the sidebar first.")
        return
    
    query = st.text_area(
        "Enter your data query:",
        value=st.session_state.last_query,
        height=100,
        help="Query"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        process_query = st.button("Process Query", type="primary")
    
    with col2:
        if st.button("Clear Results"):
            st.session_state.generated_dataframe = None
            st.session_state.last_query = ''
            st.rerun()
    
    if process_query and query and st.session_state.router_agent:
        st.session_state.last_query = query
        
        with st.spinner("Processing your query..."):
            try:
                result = st.session_state.router_agent.process(query)
                
                if result.success:
                    st.success("Query processed successfully!")
                    output = result.data.get('output', result.data) if isinstance(result.data, dict) else result.data
                    st.text_area("Query Result:", value=str(output), height=200)
                    
                    if result.metadata.get("sql_dataframe") is not None:
                        df = result.metadata["sql_dataframe"]
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            st.session_state.generated_dataframe = df
                            st.subheader("Generated DataFrame (SQL)")
                            st.dataframe(df, use_container_width=True)
                            st.info(f"DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns")
                    
                    elif result.metadata.get("csv_dataframe") is not None:
                        df = result.metadata["csv_dataframe"]
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            st.session_state.generated_dataframe = df
                            st.subheader("Generated DataFrame (CSV)")
                            st.dataframe(df, use_container_width=True)
                            st.info(f"DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns")
                    
                    if st.session_state.generated_dataframe is not None:
                        csv_buffer = BytesIO()
                        st.session_state.generated_dataframe.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="Download DataFrame as CSV",
                            data=csv_buffer.getvalue(),
                            file_name="query_result.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.error(f"Query failed: {result.error}")
                    
            except Exception as e:
                st.error(f"Error processing query: {e}")
    
    if st.session_state.generated_dataframe is not None:
        st.subheader("Current Generated DataFrame")
        st.dataframe(st.session_state.generated_dataframe, use_container_width=True)
        st.info(f"Shape: {st.session_state.generated_dataframe.shape[0]} rows × {st.session_state.generated_dataframe.shape[1]} columns")

def display_visualization_tab():
    st.header("Visualization from Generated Data")
    
    if st.session_state.generated_dataframe is None:
        st.warning("No DataFrame available. Please generate a DataFrame first in the 'DataFrame Query' tab.")
        return
    
    if not st.session_state.initialization_complete:
        st.warning("Please complete configuration and initialization in the sidebar first.")
        return
    
    st.subheader("Current DataFrame")
    with st.expander("View DataFrame", expanded=False):
        st.dataframe(st.session_state.generated_dataframe, use_container_width=True)
        st.info(f"Shape: {st.session_state.generated_dataframe.shape[0]} rows × {st.session_state.generated_dataframe.shape[1]} columns")
    
    if st.session_state.generated_dataframe is not None and st.session_state.groq_api_key:
        try:
            llm = ChatGroq(
                model_name="llama-3.1-8b-instant",
                api_key=st.session_state.groq_api_key,
                temperature=0.1
            )
            
            viz_agent = VisualizationAgent(
                llm=llm,
                df=st.session_state.generated_dataframe,
                plots_dir="streamlit_plots"
            )
            
            viz_query = st.text_area(
                "Enter your visualization request:",
                height=100,
                help="Describe what kind of chart or graph you want to create"
            )
            
            col1, col2 = st.columns([1, 4])
            
            with col1:
                create_viz = st.button("Create Visualization", type="primary")
            
            if create_viz and viz_query:
                with st.spinner("Creating visualization..."):
                    try:
                        result = viz_agent.process(viz_query)
                        
                        if result.success:
                            st.success("Visualization created successfully!")
                            output = result.data.get('output', result.data) if isinstance(result.data, dict) else result.data
                            st.text_area("Visualization Details:", value=str(output), height=150)
                            
                            if result.metadata.get("plot_path"):
                                plot_path = result.metadata["plot_path"]
                                if os.path.exists(plot_path):
                                    st.subheader("Generated Visualization")
                                    st.image(plot_path, use_container_width=True)
                                    with open(plot_path, "rb") as file:
                                        st.download_button(
                                            label="Download Visualization",
                                            data=file.read(),
                                            file_name=os.path.basename(plot_path),
                                            mime="image/png"
                                        )
                                else:
                                    st.warning("Plot file not found")
                        else:
                            st.error(f"Visualization failed: {result.error}")
                            
                    except Exception as e:
                        st.error(f"Error creating visualization: {e}")
        
        except Exception as e:
            st.error(f"Error initializing visualization agent: {e}")

def display_raw_data_visualization_tab():
    st.header("Raw Data Visualization")
    
    if st.session_state.raw_dataframe is None:
        st.warning("No raw data available. Please upload a CSV file in the sidebar.")
        return
    
    if not st.session_state.groq_api_key:
        st.warning("Please provide GROQ API Key in the sidebar.")
        return
    
    st.subheader("Raw Data Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", st.session_state.raw_dataframe.shape[0])
    with col2:
        st.metric("Columns", st.session_state.raw_dataframe.shape[1])
    
    with st.expander("Data Preview", expanded=True):
        tab1, tab2, tab3 = st.tabs(["Sample Data", "Data Types", "Summary Stats"])
        
        with tab1:
            st.dataframe(st.session_state.raw_dataframe.head(10), use_container_width=True)
        
        with tab2:
            dtype_df = pd.DataFrame({
                'Column': st.session_state.raw_dataframe.columns,
                'Data Type': st.session_state.raw_dataframe.dtypes.astype(str),
                'Non-Null Count': st.session_state.raw_dataframe.count(),
                'Null Count': st.session_state.raw_dataframe.isnull().sum()
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with tab3:
            numeric_cols = st.session_state.raw_dataframe.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.dataframe(st.session_state.raw_dataframe[numeric_cols].describe(), use_container_width=True)
            else:
                st.info("No numeric columns for summary statistics")
    
    st.subheader("Create Visualization")
    
    try:
        llm = ChatGroq(
            model_name="openai/gpt-oss-120b",
            api_key=st.session_state.groq_api_key,
            temperature=0.1
        )
        
        viz_agent = VisualizationAgent(
            llm=llm,
            df=st.session_state.raw_dataframe,
            plots_dir="streamlit_plots"
        )
        
        st.subheader("Quick Visualizations")
        
        col1, col2, col3 = st.columns(3)
        
        numeric_columns = st.session_state.raw_dataframe.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = st.session_state.raw_dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
        
        with col1:
            if st.button("Data Overview") and len(numeric_columns) > 0:
                create_quick_visualization(viz_agent, "Create a comprehensive data overview with histograms for all numeric columns")
        
        with col2:
            if st.button("Correlation Matrix") and len(numeric_columns) > 1:
                create_quick_visualization(viz_agent, "Create a correlation heatmap for all numeric variables")
        
        with col3:
            if st.button("Missing Values") and st.session_state.raw_dataframe.isnull().sum().sum() > 0:
                create_quick_visualization(viz_agent, "Create a visualization showing missing values pattern in the dataset")
        
        st.subheader("Custom Visualization")
        custom_query = st.text_area(
            "Describe the visualization you want:",
            height=100,
            help="Example: 'Create a bar chart showing the top 10 goal scorers' or 'Show the distribution of goals by team'"
        )
        
        if st.button("Generate Custom Visualization", type="primary") and custom_query:
            create_quick_visualization(viz_agent, custom_query)
    
    except Exception as e:
        st.error(f"Error setting up visualization agent: {e}")

def create_quick_visualization(viz_agent, query):
    with st.spinner(f"Creating visualization: {query}"):
        try:
            result = viz_agent.process(query)
            
            if result.success:
                st.success("Visualization created!")
                output = result.data.get('output', result.data) if isinstance(result.data, dict) else result.data
                with st.expander("Visualization Details"):
                    st.text(str(output))
                
                if result.metadata.get("plot_path"):
                    plot_path = result.metadata["plot_path"]
                    if os.path.exists(plot_path):
                        st.image(plot_path, use_container_width=True)
                        with open(plot_path, "rb") as file:
                            st.download_button(
                                label="Download Plot",
                                data=file.read(),
                                file_name=os.path.basename(plot_path),
                                mime="image/png"
                            )
                    else:
                        st.warning("Plot file not found")
            else:
                st.error(f"Visualization failed: {result.error}")
                
        except Exception as e:
            st.error(f"Error creating visualization: {e}")

def main():
    initialize_session_state()
    st.title("Intelligent Data Analysis & Visualization Platform")
    st.markdown("---")
    
    config, api_key = load_configuration()
    
    if st.sidebar.button("Initialize System", type="primary"):
        if initialize_agents(config, api_key):
            st.sidebar.success("System ready!")
        else:
            st.sidebar.error("Initialization failed!")
    
    if st.session_state.initialization_complete:
        st.sidebar.success("System Status: Ready")
    else:
        st.sidebar.warning("System Status: Not Initialized")
    
    tab1, tab2, tab3 = st.tabs([
        "DataFrame Query", 
        "Generated Data Visualization", 
        "Raw Data Visualization"
    ])
    
    with tab1:
        display_dataframe_tab()
    
    with tab2:
        display_visualization_tab()
    
    with tab3:
        display_raw_data_visualization_tab()
    
    st.markdown("---")

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


if __name__ == "__main__":
    main()
