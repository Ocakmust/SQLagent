from langchain_groq import ChatGroq
from loggerCenter import LoggerCenter
from utils import AgentResult, CodeOutput
from utils import BaseSpecializedAgent
from langchain.tools import Tool
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from vectordeneme import ContextFind  
from document import DocumentProcessor
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate


logger = LoggerCenter().get_logger()

class SQLQuerryAgent(BaseSpecializedAgent):

    def __init__(self, llm: ChatGroq, db_manager, doc_path: str = None, columnInfo_path: str = None):
        super().__init__("SQLagent", llm)

        self.parser = PydanticOutputParser(pydantic_object=CodeOutput)
        self.db_manager = db_manager  
        self.doc_path = doc_path
        self.columnInfo_path = columnInfo_path
        self.db_tables = None
        self.db_schema = None
        
        self.context_finder = None
        self.doc_process = None
        self.columnInfo = None
        self.current_context = "No data available"

        if doc_path:
            try:
                self.context_finder = ContextFind(doc_path)
                logger.info(f"Context finder initialized for: {doc_path}")
            except Exception as e:
                logger.error(f"Failed to initialize context finder: {e}")
                self.context_finder = None

        if columnInfo_path:
            try:
                self.doc_process = DocumentProcessor()
                self.columnInfo = self.doc_process.extract_text_from_documents(columnInfo_path)
                logger.info(f"Column info loaded from: {columnInfo_path}")
            except Exception as e:
                logger.error(f"Failed to load column info: {e}")
                self.columnInfo = None

    def _get_system_prompt(self):
        return """
You are an expert SQL analyst with access to database tools and documentation context.

AVAILABLE TOOLS:
1. get_database_schema_tables: Get database schema and table information
2. get_Info: Get relevant context and column info from documents based on query
3. generate_sql_query: Generate SQL query based on requirements and context
4. execute_sql_query: Execute SQL query and return results

INSTRUCTIONS:
- If documents are available, get relevant Context or Column info for understanding business requirements
- Then, get database schema and tables information
- For SQL generation: use generate_sql_query with context and schema
- For data retrieval: use execute_sql_query to run queries and get results
- Use document context to understand business rules and data relationships
- Write clean, efficient SQL queries that align with business requirements
- ALWAYS provide a clear final answer after using tools

Your goal is to help users query and analyze database data using both technical schema and business context!
"""

    def _setup_agent(self):
        """Override to use better agent executor settings"""
        try:
            prompt = self._setup_prompt()
            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            
            self.agent_executor = AgentExecutor(
                agent=agent, 
                tools=self.tools, 
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3,
                early_stopping_method="generate",
                return_intermediate_steps=True
            )
            logger.info(f"Agent {self.agent_name} setup completed successfully")
        except Exception as e:
            logger.error(f"Agent {self.agent_name} setup failed: {e}")
            raise
    
    def _is_safe_sql(self, sql_query: str) -> bool:
        """Check if SQL query is safe to execute"""
        dangerous_keywords = [
            'drop table', 'drop database', 'delete from', 'truncate',
            'alter table', 'create table', 'insert into', 'update',
            'grant', 'revoke', 'exec', 'execute', 'xp_', 'sp_',
            'drop index', 'drop view', 'drop schema'
        ]
        
        sql_lower = sql_query.lower()
        for keyword in dangerous_keywords:
            if keyword in sql_lower:
                logger.warning(f"Potentially dangerous SQL detected: {keyword}")
                return False
        return True

    def _setup_tools(self):

        def get_database_schema_tables(input_text: str = "") -> str:
            """Get Database Schema and Tables information using DatabaseManager"""
            try:
                logger.info("Getting Database Schema and Tables")
                
                if not self.db_manager.is_connected():
                    self.db_manager.connect()
            
                self.db_tables = self.db_manager.get_table_names()
                
                if self.db_tables:
                    schema_info = []
                    for table in self.db_tables[:5]:  
                        try:
                            schema = self.db_manager.get_table_schema(table)
                            columns_info = ', '.join([f"{col['column_name']} ({col['data_type']})" for col in schema])
                            schema_info.append(f"Table {table}: {columns_info}")
                        except Exception as e:
                            logger.warning(f"Could not get schema for table {table}: {e}")
                            schema_info.append(f"Table {table}: Schema unavailable")
                    
                    self.db_schema = "\n".join(schema_info)
                    
                    result = f"""DATABASE INFORMATION:
Available Tables: {', '.join(self.db_tables)}

Table Schemas:
{self.db_schema}

Total Tables: {len(self.db_tables)}
Connection Status: Connected"""
                    return result
                else:
                    return "No database tables found or connection unavailable"
            
            except Exception as e:
                error_msg = f"Get Database Schema and Tables failed: {e}"
                logger.error(error_msg)
                return error_msg

        def get_Info(user_query: str = "") -> str:
            """Get relevant context from documents and column info"""
            try:
                logger.info(f"Getting document info for query: {user_query}")
                
                # Initialize variables
                columnInfo_info = "No column info available"
                context_info = "No context available"
                doc_link = "Not provided"
                columnInfo_link = "Not provided"
                
                # Get context from document if available
                if self.context_finder:
                    try:
                        context_info = self.context_finder.return_context(user_query, top_k=3)
                        if not context_info.strip():
                            context_info = "No relevant context found"
                        doc_link = self.doc_path
                        # Store for later use
                        self.current_context = context_info
                    except Exception as e:
                        logger.error(f"Error getting context: {e}")
                        context_info = f"Error retrieving context: {str(e)}"

                # Get column info if available
                if self.columnInfo:
                    columnInfo_info = self.columnInfo
                    columnInfo_link = self.columnInfo_path
                
                # Format result
                if self.doc_path or self.columnInfo_path:
                    result = f"""RELEVANT CONTEXT:
{context_info}

COLUMN INFO:
{columnInfo_info}

Context retrieved from: {doc_link}
Column Info retrieved from: {columnInfo_link}"""
                    return result
                else:
                    return "No document sources available for additional context"
                    
            except Exception as e:
                error_msg = f"Document info retrieval failed: {e}"
                logger.error(error_msg)
                return error_msg

        def generate_sql_query(user_request: str = "") -> str:
            """Generate SQL query based on user request, database schema, and document context"""
            try:
                logger.info(f"Generating SQL query for request: {user_request}")

                
                if not self.db_tables or not self.db_schema:
                    return "Database schema not loaded. Please get database schema first using get_database_schema_tables."
                
                # Prepare context information
                context_sql = "No context available"
                columnInfo_sql = "No column info available"
                
                # Use stored context if available
                if hasattr(self, 'current_context') and self.current_context and self.current_context != "No data available":
                    context_sql = self.current_context
                
                if self.columnInfo:
                    columnInfo_sql = self.columnInfo

                sql_generation_prompt = f"""Based on the database schema, business context, and user request, generate a SQL query.

DATABASE SCHEMA:
{self.db_schema}

AVAILABLE TABLES: {', '.join(self.db_tables)}

BUSINESS CONTEXT:
{context_sql}

COLUMN INFO:
{columnInfo_sql}

USER REQUEST: {user_request}

INSTRUCTIONS:
- Generate a clean, efficient SQL query
- Use business rules from the context if available
- Ensure column names and table relationships are correct
- Return only the SQL query without explanations or markdown formatting
- Make sure the query aligns with business requirements mentioned in the context"""
                
                response = self.llm.invoke(sql_generation_prompt)
                sql_query = response.content.strip()
                
                if sql_query.startswith('```sql'):
                    sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
                elif sql_query.startswith('```'):
                    sql_query = sql_query.replace('```', '').strip()
                
                logger.info(f"Generated SQL query: {sql_query}")
                return f"Generated SQL Query:\n{sql_query}\n\nBased on request: {user_request}"
                
            except Exception as e:
                error_msg = f"SQL generation failed: {e}"
                logger.error(error_msg)
                return error_msg

        def execute_sql_query(sql_query: str = "") -> str:
            """Execute SQL query safely using DatabaseManager and return results"""
            try:
                logger.info(f"Executing SQL query: {sql_query}")
                
                if not self.db_manager.is_connected():
                    self.db_manager.connect()
                
                sql_query = sql_query.strip()
               
                if sql_query.startswith('```sql'):
                    sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
                elif sql_query.startswith('```'):
                    sql_query = sql_query.replace('```', '').strip()
                
                if not self._is_safe_sql(sql_query):
                    return "SQL query contains potentially dangerous operations and cannot be executed"
                
                result_data = self.db_manager.execute_query(sql_query)
                
                if result_data:
                    result_df = pd.DataFrame(result_data)
                    
                    num_rows = len(result_df)
                    num_cols = len(result_df.columns)
                    
                    if num_rows > 20:
                        display_df = result_df.head(20)
                        result_text = f"""SQL Executed Successfully!

Query: {sql_query}

Results ({num_rows} rows, {num_cols} columns - showing first 20):
{display_df.to_string(index=False)}"""
                    else:
                        result_text = f"""SQL Executed Successfully!

Query: {sql_query}

Results ({num_rows} rows, {num_cols} columns):
{result_df.to_string(index=False)}"""
                    
                    logger.info(f"Query executed successfully, returned {num_rows} rows")
                    return result_text
                    
                else:
                    if sql_query.strip().upper().startswith('SELECT'):
                        return f"SQL Executed Successfully!\n\nQuery: {sql_query}\n\nResults: No data returned (empty result set)"
                    else:
                        return f"SQL Executed Successfully!\n\nQuery: {sql_query}\n\nResults: Query executed successfully"
                
            except Exception as e:
                error_msg = f"SQL execution failed: {e}\n\nQuery: {sql_query}"
                logger.error(error_msg)
                return error_msg

        self.tools = [
            Tool(
                name="get_database_schema_tables", 
                description="Get database schema and table information using DatabaseManager", 
                func=get_database_schema_tables
            ),
            Tool(
                name="get_Info",
                description="Get relevant business context and column info from documents based on user query",
                func=get_Info
            ),
            Tool(
                name="generate_sql_query",
                description="Generate SQL query based on user request, database schema, and business context",
                func=generate_sql_query
            ),
            Tool(
                name="execute_sql_query",
                description="Execute SQL query safely using DatabaseManager and return formatted results",
                func=execute_sql_query
            )
        ]
    
    def process(self, query: str = None) -> AgentResult:
        """Process query with better output handling"""
        if query:
            self.query = query
            
        logger.info(f"Processing SQL query: {self.query}")

        try:
            response = self.agent_executor.invoke({"input": self.query})
            
            # Handle malformed output like <function=get>
            output = None
            if isinstance(response, dict):
                if 'output' in response and response['output'] and response['output'] != '<function=get>':
                    output = response['output']
                elif 'intermediate_steps' in response and response['intermediate_steps']:
                    # Extract from intermediate steps if output is malformed
                    for step in response['intermediate_steps']:
                        if len(step) > 1 and isinstance(step[1], str) and step[1] != '<function=get>':
                            output = step[1]
                            break
            
            # If we have valid output, use it
            if output:
                logger.info("SQL agent processing completed successfully")
                return AgentResult(
                    success=True,
                    data={'output': output},
                    metadata={
                        "agent": self.agent_name,
                        "query": self.query,
                        "db_tables": self.db_tables,
                        "doc_path": self.doc_path,
                        "columnInfo_path": self.columnInfo_path,
                        "context_used": bool(self.current_context and self.current_context != "No data available")
                    }
                )
            else:
                # If no valid output, return the original response
                logger.warning("Agent output may be malformed, returning original response")
                return AgentResult(
                    su