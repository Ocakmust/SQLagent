import os
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.tools import StructuredTool
from utils.document import DocumentProcessor


from utils.loggerCenter import LoggerCenter

from utils.base_agent import AgentResult, BaseSpecializedAgent
from oldagents.pandas_agent import DataAnalysisAgent
from oldagents.db_agent import SQLQuerryAgent
from api_agent import ExternalAPIAgent
from utils.db_dao import DatabaseManager
from utils.vectordeneme import ContextFind

logger = LoggerCenter().get_logger()

class RouterAgent(BaseSpecializedAgent):
    """Router agent"""
    
    def __init__(self, llm: ChatGroq, config: Dict[str, Any]):
        self.config = config
        self.sub_agents = {}
        
        self.dataframe = None
        self.db_manager = None
        self.context_finder = None
        self.column_info = None
        
        self.sql_dataframe = None
        self.csv_dataframe = None
        
        super().__init__("RouterAgent", llm)
        self.original_query = None
        self._load_resources()
        self._initialize_agents(llm)
    
    def _load_resources(self):

        self.dataframe = None
        csv_path = self.config.get('csv_path')
        if csv_path and Path(csv_path).exists():
            try:
                self.dataframe = pd.read_csv(csv_path)
                logger.info(f"CSV loaded: {self.dataframe.shape}")
            except Exception as e:
                logger.error(f"CSV loading failed: {e}")
        
        self.db_manager = None
        if self.config.get('db_params'):
            try:
                self.db_manager = DatabaseManager(self.config['db_params'])
                self.db_manager.connect()
                logger.info("Database connected")
            except Exception as e:
                logger.error(f"Database error: {e}")
        
        self.context_finder = None
        doc_path = self.config.get('doc_path')
        if doc_path and Path(doc_path).exists():
            try:
                self.context_finder = ContextFind(doc_path)
                logger.info("PDF context ready")
            except Exception as e:
                logger.error(f"PDF context error: {e}")
        
        self.column_info = None
        column_info_path = self.config.get('column_info')
        if column_info_path and Path(column_info_path).exists():
            try:
                doc_processor = DocumentProcessor()
                self.column_info = doc_processor.extract_text_from_documents(column_info_path)
                logger.info("Column info loaded")
            except Exception as e:
                logger.error(f"Column info loading failed: {e}")
    
    def _initialize_agents(self, llm: ChatGroq):
        """sub agentları başlat"""
        try:
            if self.dataframe is not None:
                self.sub_agents['data'] = DataAnalysisAgent(
                    llm=llm,
                    df=self.dataframe,
                    doc_path=self.config.get('doc_path'),
                    column_info_path=self.config.get('column_info')
                    )
                logger.info("Data Analysis Agent ready")
            
            if self.db_manager is not None:
                self.sub_agents['sql'] = SQLQuerryAgent(
                    llm=llm,
                    db_manager=self.db_manager,
                    doc_path=self.config.get('doc_path'),
                    columnInfo_path=self.config.get('column_info')
                )
                logger.info("SQL Agent ready")
            
            self.sub_agents['api'] = ExternalAPIAgent(llm=llm)
            logger.info("API Agent ready")
            
        except Exception as e:
            logger.error(f"Agent initialization error: {e}")

    def _get_available_resources_summary(self) -> str:
        """prompt için özet bilgiler """
        resources = []
        
        if self.dataframe is not None:
            columns = list(self.dataframe.columns)[:20]
            resources.append(f"Info from CSV Data: {self.dataframe.shape[0]} rows, Columns: {', '.join(columns)}")
        
        if self.db_manager:
            try:
                tables = self.db_manager.get_table_names()
            
                schema_info = []
                for table in tables[:5]:  
                    try:
                        schema = self.db_manager.get_table_schema(table)
                        columns_info = ', '.join([f"{col['column_name']} ({col['data_type']})" for col in schema])
                        schema_info.append(f"Table {table}: {columns_info}")
                    except Exception as e:
                        logger.warning(f"Could not get schema for table {table}: {e}")
                        schema_info.append(f"Table {table}: Schema unavailable")
                
                available_schema = "\n".join(schema_info)
                available_tables = ', '.join(tables)
                
                db_info=f"""
DATABASE SCHEMA:
Available Tables: 
{available_tables}

Table Schemas:
{available_schema}
"""

                if tables:
                    resources.append(f"Database: {db_info}")
                else:
                    resources.append("Database: Connected but no tables found")
            except:
                resources.append("Database: Connected")
        
        if self.column_info:
            resources.append("Column infos are available.")
        
        resources.append("External APIs: Weather, news available")
        
        return "\n".join(resources) if resources else "No resources loaded"


    def _get_system_prompt(self) -> str:
        """Basit sistem prompt'u"""
        return f"""You are a router that directs user questions to the right agent.

AVAILABLE RESOURCES:
{self._get_available_resources_summary()}

TOOLS:
1. route_to_data - For CSV data analysis and statistics
2. route_to_sql - For database queries
3. route_to_api - For weather, news, external info

SIMPLE ROUTING RULES:
- If question is about CSV data → use route_to_data
- If question is about database/tables → use route_to_sql  
- If question is about weather/news/current info → use route_to_api

IMPORTANT RULES FOR ROUTING:
1. For each user query, you MUST choose exactly ONE tool to execute. Do NOT attempt to call multiple tools, chain tool calls, or loop between tools.
2. Always pass the ORIGINAL user query to the selected tool without any modifications, explanations, or comments.
3. Once a tool returns a result, immediately use that result to provide the final answer to the user. Do NOT attempt to call another tool afterward.
4. Do NOT generate any additional queries or attempt to guess what other tools might be useful. Stick strictly to the single tool chosen.
5. Never enter a loop between tools. Even if the output suggests further queries, you must stop after one tool execution.

NOTE:
- Use the context of the user query only to select the appropriate tool. Do not modify the query for the tool.
- If a query is ambiguous, pick the best single tool based on the available resources and rules above.
"""
    def _setup_tools(self):
        """Router araçlarını ayarla"""
        
        class DataRoutingInput(BaseModel):
            query: str = Field(description="Original user query for data analysis")
        
        class SQLRoutingInput(BaseModel):
            query: str = Field(description="Original user query for database operations")
        
        class APIRoutingInput(BaseModel):
            query: str = Field(description="Original user query for external APIs")

 

        def route_to_data(query: str) -> str:
            """Route to data analysis agent"""
            if 'data' not in self.sub_agents:
                return "CSV data not available"
            
            try:
                actual_query = self.original_query if self.original_query else query
                logger.info(f"Routing to data analysis: {actual_query}")
                result = self.sub_agents['data'].process(actual_query)
                
                if result.success:
                    if hasattr(result, 'metadata') and result.metadata is not None:
                        df = result.metadata.get('dataframe')
                        if df is not None and isinstance(df, pd.DataFrame):
                            self.csv_dataframe = df

                    output = result.data.get('output', result.data) if isinstance(result.data, dict) else result.data
                    return str(output)
                else:
                    return f"Data analysis error: {result.error}"
            except Exception as e:
                return f"Error: {str(e)}"

        def route_to_sql(query: str) -> str:
            """Route to SQL agent"""
            if 'sql' not in self.sub_agents:
                return "Database not available"
            
            try:
                actual_query = self.original_query if self.original_query else query
                logger.info(f"Routing to SQL: {actual_query}")
                result = self.sub_agents['sql'].process(actual_query)
                
                if result.success:
                    if hasattr(result, 'metadata') and result.metadata is not None:
                        df = result.metadata.get("dataframe")
                        if df is not None and isinstance(df, pd.DataFrame):
                            self.sql_dataframe = df
                    
                    output = result.data.get('output', result.data) if isinstance(result.data, dict) else result.data
                    return str(output)
                else:
                    return f"SQL error: {result.error}"
            except Exception as e:
                return f"Error: {str(e)}"

        def route_to_api(query: str) -> str:
            """Route to API agent"""
            if 'api' not in self.sub_agents:
                return "API agent not available"
            
            try:
                actual_query = self.original_query if self.original_query else query
                logger.info(f"Routing to API: {actual_query}")
                result = self.sub_agents['api'].process(actual_query)
                
                if result.success:
                    return str(result.data)
                else:
                    return f"API error: {result.error}"
            except Exception as e:
                return f"Error: {str(e)}"

        self.tools = [
            StructuredTool.from_function(
                name="route_to_data",
                description="Use for CSV data analysis, statistics, charts. Pass the original user query without modifications.",
                func=route_to_data,
                args_schema=DataRoutingInput
            ),
            StructuredTool.from_function(
                name="route_to_sql", 
                description="Use for database queries and SQL operations. Pass the original user query without modifications.",
                func=route_to_sql,
                args_schema=SQLRoutingInput
            ),
            StructuredTool.from_function(
                name="route_to_api",
                description="Use for weather, news, external information. Pass the original user query without modifications.",
                func=route_to_api,
                args_schema=APIRoutingInput
            )
        ]

    def process(self, query: str) -> AgentResult:
        """Process user query"""
        logger.info(f"Router processing query: {query}")
        
        try:
            # Store original query for sub-agents
            self.original_query = query
            
            response = self.agent_executor.invoke({"input": query})
            
            return AgentResult(
                success=True,
                data=response,
                metadata={
                    "agent": self.agent_name,
                    "original_query": query,
                    "sql_dataframe": self.sql_dataframe,
                    "csv_dataframe": self.csv_dataframe,
                }
            )
            
        except Exception as e:
            error_msg = f"Router error: {str(e)}"
            logger.error(error_msg)
            return AgentResult(success=False, error=error_msg)

def main():
    """Main application"""
    print("Intelligent Router System")
    print("=" * 40)
    
    config = {
        'db_params': {
            "host": "localhost",
            "database": "musteri_db", 
            "user": "postgres",
            "password": "123",
            "port": "5432"
        },
        'csv_path':"Data/goalscorers.csv"
    }
    
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found")
        
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            api_key=groq_api_key,
            temperature=0.1
        )
        
        print("Initializing router...")
        router = RouterAgent(llm=llm, config=config)
        print("Router ready!\n")
        
        try:
            query = "Get me a list of 10 scorers who scored most goals and their total goals. Data must have columns of scorers and the total goals they have scored."
            
            
            print(f"\nProcessing: {query}")
            result = router.process(query)
            
            if result.success:
                print("Result:")
                output = result.data.get('output', result.data) if isinstance(result.data, dict) else result.data
                print(output)
                
                # Check for DataFrames
                if result.metadata.get("sql_dataframe") is not None:
                    sql_df = result.metadata["sql_dataframe"] 
                    if isinstance(sql_df, pd.DataFrame) and not sql_df.empty:
                        print(f"\nSQL DataFrame available: {sql_df.shape}")
                        print(sql_df.head())
                
                if result.metadata.get("csv_dataframe") is not None:
                    csv_df = result.metadata["csv_dataframe"]
                    if isinstance(csv_df, pd.DataFrame) and not csv_df.empty:
                        print(f"\nCSV DataFrame available: {csv_df.shape}")
                        print(csv_df.head())
                    
            else:
                print(f"Error: {result.error}")
            
            print("\n" + "-" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"Error: {e}")
                
    except Exception as e:
        print(f"System initialization failed: {e}")

if __name__ == "__main__":
    main()