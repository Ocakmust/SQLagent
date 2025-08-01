from typing import Optional
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
from langchain_core.prompts import PromptTemplate
from langchain.tools import StructuredTool



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
                self.columnInfo = self.doc_process.extract_text_from_documents(columnInfo_path)  # Fixed method name
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
- ALWAYS GET relevant Context or Column info for understanding business requirements
- Then, get database schema and tables information
- For SQL generation: use generate_sql_query with context and schema
- For data retrieval: use execute_sql_query to run queries and get results
- Use document context to understand business rules and data relationships
- Write clean, efficient SQL queries that align with business requirements

Your goal is to help users query and analyze database data using both technical schema and business context!
"""
    
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

    def _generating_sql_query(self, user_request: str) -> str:
        """Generate SQL query for the given request"""
        template = """
    You are an expert SQL analyst. You analyze the request STEP BY STEP and generate a clean SQL query that answers the request using the provided database schema and business context.

    IMPORTANT:
    1. Do NOT include anything outside of SQL CODE and an Explanation.
    2. Explanation MUST be about what is understood from the REQUEST.
    Your response must be exactly this format:
    {{"code": "your_sql_query_here", "explanation": "brief explanation"}}

    {format_instructions}

    CRITICAL REQUIREMENTS:
    1. Generate ONLY the SQL query without any markdown formatting or code blocks
    2. Use proper table and column names from the schema
    3. UNDERSTAND what the request wants exactly
    4. Follow SQL best practices and optimization
    5. Ensure the query is safe (SELECT operations only for data retrieval)
    6. Use business context to understand data relationships and requirements

    DATABASE SCHEMA:
    {db_schema}

    AVAILABLE TABLES: 
    {available_tables}

    BUSINESS CONTEXT:
    {business_context}

    COLUMN INFO:
    {column_info}

    Request: {user_request}
    """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["user_request", "db_schema", "available_tables", "business_context", "column_info"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        chain = prompt | self.llm | self.parser
        
        logger.info(f"Generating SQL query for request: {user_request}")
                        
        try:
            business_context = "No context available"
            column_info = "No column info available"
            
            if hasattr(self, 'current_context') and self.current_context and self.current_context != "No data available":
                business_context = self.current_context
            
            if self.columnInfo:
                column_info = self.columnInfo

            response = chain.invoke({
                "user_request": user_request,
                "db_schema": self.db_schema or "Schema not loaded",
                "available_tables": ', '.join(self.db_tables) if self.db_tables else "No tables available",
                "business_context": business_context,
                "column_info": column_info
            })
            
            if isinstance(response, dict):
                if "text" in response:
                    sql_query = response["text"].code if hasattr(response["text"], 'code') else str(response["text"])
                else:
                    sql_query = response.get("code", str(response))
            else:
                sql_query = response.code if hasattr(response, 'code') else str(response)
            
            sql_query = sql_query.strip()
            if sql_query.startswith('```sql'):
                sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            elif sql_query.startswith('```'):
                sql_query = sql_query.replace('```', '').strip()
            
            logger.info(f"SQL query generated: {sql_query}")
            return f"{sql_query}"
            
        except Exception as e:
            logger.error(f"Error in SQL generation: {e}")
            return f"Error generating SQL query: {str(e)}"


    def _setup_tools(self):

        class DatabaseSchemaInput(BaseModel):
            input_text: Optional[str] = Field(default="", description="Optional input text")
        
        class InfoInput(BaseModel):
            user_query: str = Field(description="User query to find relevant context")
        
        class SQLGenerationInput(BaseModel):
            user_request: str = Field(description="User request for SQL query generation")
        
        class SQLExecutionInput(BaseModel):
            sql_query: str = Field(description="SQL query to execute")

        def get_database_schema_tables(input_text: str) -> str:
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

        def get_Info(user_query: str) -> str:
            """Get relevant context from documents and column info"""
            try:
                logger.info(f"Getting document info for query: {user_query}")
                
                columnInfo_info = "No column info available"
                context_info = "No context available"
                doc_link = "Not provided"
                columnInfo_link = "Not provided"
                
                if self.context_finder:
                    try:
                        context_info = self.context_finder.return_context(user_query, top_k=3)
                        if not context_info.strip():
                            context_info = "No relevant context found"
                        doc_link = self.doc_path
                        self.current_context = context_info
                    except Exception as e:
                        logger.error(f"Error getting context: {e}")
                        context_info = f"Error retrieving context: {str(e)}"

                if self.columnInfo:
                    columnInfo_info = self.columnInfo
                    columnInfo_link = self.columnInfo_path
                
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

        def generate_sql_query(user_request: str) -> str:
            """Generate SQL query based on user request, database schema, and document context"""
            try:
                logger.info(f"Generating SQL query for request: {user_request}")
                
                result = self._generating_sql_query(user_request)
                return result
                
            except Exception as e:
                error_msg = f"SQL generation failed: {e}"
                logger.error(error_msg)
                return error_msg

        def execute_sql_query(sql_query: str) -> str:
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
{display_df.to_string(index=False)}

"""
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
            StructuredTool.from_function(
                name="get_database_schema_tables", 
                description="Get database schema and table information using DatabaseManager", 
                func=get_database_schema_tables,
                args_schema=DatabaseSchemaInput

            ),
            StructuredTool.from_function(
                name="get_Info",
                description="Get relevant business context and column info from documents based on user query",
                func=get_Info,
                args_schema=InfoInput
            ),
            StructuredTool.from_function(
                name="generate_sql_query",
                description="Generate SQL query based on user request, database schema, and business context",
                func=generate_sql_query,
                args_schema=SQLGenerationInput
            ),
            StructuredTool.from_function(
                name="execute_sql_query",
                description="Execute SQL query safely using DatabaseManager and return formatted results",
                func=execute_sql_query,
                args_schema=SQLExecutionInput
            )
        ]
    
    def process(self, query: str = None) -> AgentResult:
        """Process query through SQL agent with document context and database operations"""
        if query:
            self.query = query
            
        logger.info(f"Processing SQL query: {self.query}")

        try:
            response = self.agent_executor.invoke({"input": self.query})
            
            logger.info("SQL agent processing completed successfully")
            return AgentResult(
                success=True,
                data=response,
                metadata={
                    "agent": self.agent_name,
                    "query": self.query,
                    "db_tables": self.db_tables,
                    "doc_path": self.doc_path,
                    "columnInfo_path": self.columnInfo_path,
                    "context_used": bool(self.current_context and self.current_context != "No data available")
                }
            )
            
        except Exception as e:
            error_msg = f"SQL agent processing failed: {str(e)}"
            logger.error(error_msg)
            return AgentResult(
                success=False,
                error=error_msg,
                metadata={
                    "agent": self.agent_name,
                    "query": self.query
                }
            )

def example_usage():
    """Example usage with corrected parameters"""
    import os
    from db_dao import DatabaseManager
    
    db_params = {
        "host": "localhost",
        "database": "musteri_db",
        "user": "postgres", 
        "password": "123",
        "port": "5432"
    }
    
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            print("❌ GROQ_API_KEY not found")
            return
        
        llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=groq_api_key)
        db_manager = DatabaseManager(db_params)
        
        # Note: Fixed parameter order - db_manager comes first
        agent = SQLQuerryAgent(
            llm=llm,
            db_manager=db_manager,
            columnInfo_path="file.pdf"
        )
        
        test_queries = [
            "Get me top 5 rows having highest EXIST KRS BNK NONCASH RISK value. You can find "
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing: {query}")
            result = agent.process(query)
            
            if result.success:
                print(f"✅ Success")
                print(f"📋 Result: {result.data}")
            else:
                print(f"❌ Error: {result.error}")
            print("-" * 50)
            
    except Exception as e:
        print(f"❌ Example failed: {e}")

if __name__ == "__main__":
    example_usage()