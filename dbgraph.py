from typing import Any, Dict, Literal, Optional, TypedDict
from langchain_groq import ChatGroq
from oldagents.pandas_agent import CodeOutput
from utils.loggerCenter import LoggerCenter
from utils.base_agent import BaseSpecializedAgent,AgentResult
from langchain.tools import Tool
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from utils.vectordeneme import ContextFind  
from utils.document import DocumentProcessor
from langchain_core.prompts import PromptTemplate
from langchain.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from enum import Enum
import json
logger = LoggerCenter().get_logger()


class SQLExecutionResult(BaseModel):
    success: bool
    query: str
    dataframe: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None
    formatted_output: str = ""

    class Config:
        arbitrary_types_allowed = True

class WorkflowType(str, Enum):
    INFO_ONLY = "info_only"
    GENERATE_QUERY_ONLY = "generate_query_only"
    EXECUTE_QUERY = "execute_query"
    DIRECT_EXECUTE = "direct_execute"
    INVALID = "invalid"

class SQLRoutingDecision(BaseModel):
    """Structured routing decision with validation"""
    workflow_type: WorkflowType = Field(description="The type of workflow to execute")
    reasoning: str = Field(description="Explanation of the routing decision")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in routing decision")

class ErrorCorrection(BaseModel):
    next_node: str = Field(description="Next node to execute: 'get_database_info', 'generate_sql_query', or 'END'")
    corrected_user_query: Optional[str] = Field(description="Enhanced user query that includes error context and fix instructions")
    reasoning: str = Field(description="Explanation of error analysis and query enhancement strategy")
    confidence: float = Field(description="Confidence level between 0.0 and 1.0")

class ValidationResult(BaseModel):
    is_correct: bool = Field(description="Whether the result correctly answers the user query")
    accuracy_score: float = Field(ge=0.0, le=1.0, description="Accuracy score of the result (0.0 to 1.0)")
    feedback: str = Field(description="Detailed feedback about what's wrong or missing")
    improvement_suggestions: str = Field(description="Specific suggestions to improve the query/SQL")
    should_retry: bool = Field(description="Whether the query should be retried with improvements")

class QueryEnhancement(BaseModel):
    enhanced_query: str = Field(description="Enhanced query with improvement notes and specific guidance")
    enhancement_reasoning: str = Field(description="Explanation of what was enhanced and why")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the enhancement")

class SQLState(TypedDict):
    query: str
    route_decision: Optional[SQLRoutingDecision]
    routing_reasoning: Optional[str]
    confidence_score: Optional[float]
    result: Optional[str]
    error: Optional[str]
    metadata: Optional[Dict[str, Any]]
    sql_dataframe: Optional[pd.DataFrame]
    database_info_loaded: Optional[bool]
    database_info: Optional[str]
    generated_query: Optional[str]
    execution_result: Optional[str]
    next_error_node: Optional[str]
    retry_count: Optional[int]
    validation_attempts: Optional[int]  # Track validation attempts
    validation_result: Optional[ValidationResult]
    is_validated: Optional[bool]
    final_result: Optional[str]


class SQLQuerryAgent(BaseSpecializedAgent):

    def __init__(self, llm: ChatGroq, db_manager, doc_path: str = None, columnInfo_path: str = None):
        super().__init__("SQLagent", llm)

        self.parser = PydanticOutputParser(pydantic_object=CodeOutput)
        self.db_manager = db_manager  
        self.doc_path = doc_path
        self.columnInfo_path = columnInfo_path
        self.max_retries = 5
        self.max_validation_attempts = 4
        self.db_tables = None
        self.db_schema = None

        self.context_finder = None
        if doc_path:
            try:
                self.context_finder = ContextFind(doc_path)
                logger.info(f"Context finder initialized for: {doc_path}")
            except Exception as e:
                logger.error(f"Failed to initialize context finder: {e}")
                self.context_finder = None

        self.doc_process = None
        self.columnInfo = None
        if columnInfo_path:
            try:
                self.doc_process = DocumentProcessor()
                self.columnInfo = self.doc_process.extract_text_from_documents(columnInfo_path)
                logger.info(f"Column info loaded from: {columnInfo_path}")
            except Exception as e:
                logger.error(f"Failed to load column info: {e}")
                self.columnInfo = None

        self.current_context = "No data available"
        self.the_answer = None
        self.database_info = None
        self.app = self._build_graph()

    def _llm_routing_node(self, state: SQLState) -> SQLState:
        query = state["query"]
        logger.info(f"Making intelligent LLM-based routing decision for: {query}")
            
        parser = PydanticOutputParser(pydantic_object=SQLRoutingDecision)
        
        routing_prompt = f"""
You are an intelligent routing system for SQL database operations.
Your job is to analyze the USER QUERY and decide which database operations should be executed.

USER QUERY:
"{query}"

AVAILABLE TOOLS/AGENTS:
1. get_database_info  →  Retrieves schema, tables, context, and column information
2. generate_sql_query →  Generates an SQL query from user requirements and context
3. execute_sql_query  →  Executes an SQL query and returns structured results

ROUTING RULES:
1. If the user asks for database details/schema/info → classify as "info_only"
2. If the user only wants an SQL query → classify as "generate_query_only"  
3. If the user wants both query generation and execution → classify as "execute_query"
4. If the user provides an SQL query and asks to run it → classify as "direct_execute"
5. If the user query is not related to database operations → classify as "invalid"

WORKFLOW:
- "info_only": get_database_info → END
- "generate_query_only": get_database_info → generate_sql_query → END
- "execute_query": get_database_info → generate_sql_query → execute_sql_query → END
- "direct_execute": execute_sql_query → END
- "invalid": → END

OUTPUT FORMAT:
You must output only valid JSON matching the schema below.
Do not include explanations, markdown, or code blocks.
Output JSON only.

EXAMPLE (user asks for customer analysis):
{{
    "workflow_type": "execute_query",
    "reasoning": "User wants database analysis results, need to load schema, generate SQL, and execute",
    "confidence": 0.9
}}

{parser.get_format_instructions()}
"""

        try:
            response = self.llm.invoke(routing_prompt)
            routing_decision = parser.parse(response.content)

            if routing_decision.workflow_type == WorkflowType.INVALID:
                return {
                    **state,
                    "route_decision": routing_decision,
                    "error": "Routing classification failed: Query is not suitable for database operations."
                }
            
            return {
                **state,
                "route_decision": routing_decision,
                "routing_reasoning": routing_decision.reasoning,
                "confidence_score": routing_decision.confidence
            }
            
        except Exception as e:
            logger.error(f"LLM routing error: {e}")
            fallback_decision = SQLRoutingDecision(
                workflow_type=WorkflowType.INVALID,
                reasoning=f"Routing classification failed: Error: {e}",
                confidence=0.8
            )
            
            return {
                **state,
                "route_decision": fallback_decision,
                "error": f"Routing classification failed: Error: {e}"
            }

    def _is_safe_sql(self, sql_query: str) -> bool:
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

    def get_database_info(self, state: SQLState) -> SQLState:
        try:
            logger.info("Finding database info.")
            available_tables = "No data available"
            available_schema = "No data available"
            total_tables = "No data available"

            columnInfo_info = "No column info available"
            columnInfo_link = "Not provided"

            context_info = "Context is not provided"
            doc_link = "Not provided"

            if not self.db_manager.is_connected():
                self.db_manager.connect()
        
            if self.db_manager.is_connected():
                try:
                    self.db_tables = self.db_manager.get_table_names()
                
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
                    available_tables = ', '.join(self.db_tables)
                    available_schema = self.db_schema
                    total_tables = len(self.db_tables)

                except Exception as e:
                    logger.error(f"Error getting data tables: {e}")
                    
            if self.context_finder and state["query"]:
                try:
                    context_info = self.context_finder.return_context(state["query"], top_k=3)
                    if not context_info.strip():
                        context_info = "No relevant context found"
                    doc_link = self.doc_path
                    self.current_context = context_info
                except Exception as e:
                    logger.error(f"Error getting context: {e}")

            if self.columnInfo:
                try:
                    columnInfo_info = self.columnInfo
                    columnInfo_link = self.columnInfo_path
                except Exception as e:
                    logger.error(f"Error getting column info: {e}")
            
            self.database_info = f"""
DATABASE SCHEMA:
Available Tables: {available_tables}

Table Schemas:
{available_schema}

Total Tables: {total_tables}
Connection Status: Connected

RELEVANT CONTEXT OF THE DATABASE:
{context_info}

COLUMN INFO:
{columnInfo_info}

Context retrieved from: {doc_link}
Column Info retrieved from: {columnInfo_link}"""
                    
            state["database_info_loaded"] = True
            state["database_info"] = self.database_info
            return state

        except Exception as e:
            state["error"] = f"Database info retrieval failed: {e}"
            logger.error(state["error"])
            return state

    def generate_sql_query(self, state: SQLState) -> SQLState:
        try:
            query = state["query"]
            logger.info(f"SQL query is being generated. User query -> {query}")

            if not state["database_info_loaded"]:
                logger.info("Database info not loaded, loading automatically...")
                self.get_database_info(state)
            
            parser = PydanticOutputParser(pydantic_object=CodeOutput)
            sql_prompt = f"""
You are an expert SQL analyst. Generate clean SQL query that answers the request using the loaded database information.

=== LOADED DATABASE INFORMATION ===
{self.database_info}

=== INSTRUCTIONS ===
1. Use the database schema and context information provided above
2. Pay special attention to the business context to understand the requirements
3. Do NOT include anything outside of SQL CODE and an Explanation
4. Your response must be EXACTLY one valid JSON object, no explanations, no markdown, no extra text

CRITICAL JSON OUTPUT FORMAT:
{{
  "code": "your_sql_query_here",
}}

{parser.get_format_instructions}

CRITICAL REQUIREMENTS:
1. Generate ONLY SQL query without markdown formatting
2. Use proper table and column names from the loaded schema above
3. Follow SQL best practices and optimization
4. Ensure the query is safe (SELECT operations only)
5. Database type: PostgreSQL - use appropriate syntax
6. Do NOT include any import statements or connection code

User Request: {query}
"""
            
            raw_response = self.llm.invoke(sql_prompt)
            response = parser.parse(raw_response.content)
           
            try:
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
                state["generated_query"] = sql_query
                return state
                
            except Exception as e:
                state["error"] = f"SQL query generation failed: Error while cleaning response {e}"
                logger.error(state["error"])
                return state
                                
        except Exception as e:
            state["error"] = f"SQL query generation failed: Error while producing response {e}"
            logger.error(state["error"])
            return state

    def execute_sql_query(self, state: SQLState) -> SQLState:
        
        if not state["generated_query"] and state["route_decision"].workflow_type != WorkflowType.DIRECT_EXECUTE:
            logger.info("SQL query not generated, generating automatically...")
            self.generate_sql_query(state)
        
        if state["route_decision"].workflow_type == WorkflowType.DIRECT_EXECUTE:
            state["generated_query"] = state["query"]
        
        try:
            sql_query = state["generated_query"]
            logger.info(f"Executing the SQL query.")

            if not self.db_manager.is_connected():
                self.db_manager.connect()
            
            if not self._is_safe_sql(sql_query):
                state["error"] = "SQL query contains potentially dangerous operations and cannot be executed"
                logger.error(state["error"])
                return state

            result_data = self.db_manager.execute_query(sql_query)
            
            if result_data:
                result_df = pd.DataFrame(result_data)
                
                num_rows = len(result_df)
                num_cols = len(result_df.columns)

                if result_df.empty:
                    result_text = f"""SQL Executed Successfully!

Query: {sql_query}

Result: Empty DataFrame"""
                    
                    state["execution_result"] = result_df.to_json(orient='records')
                    self.the_answer = SQLExecutionResult(
                        success=True,
                        query=sql_query,
                        dataframe=result_df,
                        formatted_output=result_text
                    )
                    return state

                if num_rows > 20:
                    display_result = result_df.head(20)
                    result_text = f"""SQL Executed Successfully!

Query: {sql_query}

Result (showing first 20 rows of {num_rows}):
{display_result.to_string(index=False)}"""
                else:
                    result_text = f"""SQL Executed Successfully!

Query: {sql_query}

Result ({num_rows} rows):
{result_df.to_string(index=False)}"""
                
                state["execution_result"] = result_df.to_json(orient='records')
                self.the_answer = SQLExecutionResult(
                    success=True,
                    query=sql_query,
                    dataframe=result_df,
                    formatted_output=result_text
                )

                logger.info(f"SQL executed successfully, returned {result_df.head()}")
                return state
                
            else:
                result_text = f"""SQL Executed Successfully!

Query: {sql_query}

Result: Query executed successfully but returned no data"""
                
                state["execution_result"] = pd.DataFrame().to_json(orient='records')
                self.the_answer = SQLExecutionResult(
                    success=True,
                    query=sql_query,
                    dataframe=pd.DataFrame(),
                    formatted_output=result_text
                )
                logger.info(f"SQL executed successfully, returned no data")
                return state

        except Exception as e:
            import traceback
            error_msg = f"SQL execution failed: {str(e)}\nTraceback:\n{traceback.format_exc()}"
            state["error"] = error_msg
            logger.error(state["error"])
            return state

    def validate_result(self, state: SQLState) -> SQLState:
        
        try:
            original_query = state.get("query", "")
            generated_query = state.get("generated_query", "")
            execution_result = state.get("execution_result", "")
            validation_attempts = state.get("validation_attempts", 0)
            
            logger.info(f"Validating SQL result (attempt {validation_attempts + 1}/{self.max_validation_attempts})")

            if validation_attempts >= self.max_validation_attempts:
                return state
            
            parser = PydanticOutputParser(pydantic_object=ValidationResult)
            
            validation_prompt = f"""
You are a code validator for SQL queries. Evaluate if the SQL CODE logically produces what the user requested.

=== ORIGINAL USER QUERY ===
"{original_query}"

=== GENERATED SQL CODE ===
{generated_query}

=== EXECUTION RESULT ===
{execution_result}

=== DATABASE CONTEXT ===
{self.database_info if self.database_info else "No database info available"}

=== SQL VALIDATION CRITERIA ===
1. Does the SQL logically address the user query?
2. Are the correct tables and column names used?
3. Is the JOIN / WHERE / GROUP BY logic appropriate?
4. Are ORDER BY, LIMIT, and aggregations correctly applied if needed?
5. Is the SQL syntax valid for PostgreSQL?

=== ACCURACY SCORING GUIDE FOR SQL ===
- 1.0: Perfect SQL
- 0.8-0.9: Good with minor issues
- 0.6-0.7: Partial with gaps
- 0.4-0.5: Poor with major issues
- 0.0-0.3: Wrong approach

=== CRITICAL RULES ===
- If accuracy_score >= 0.8, set is_correct = true and should_retry = false
- If accuracy_score < 0.8, set is_correct = false and should_retry = true
- Focus only on SQL logic, correctness of schema usage, and execution validity
- Provide concrete feedback about SQL problems and how to fix them

REQUIRED JSON OUTPUT (no additional text):
{{
    "is_correct": boolean,
    "accuracy_score": float_between_0_and_1,
    "feedback": "detailed_feedback_about_sql_logic_and_schema_usage",
    "improvement_suggestions": "specific_sql_fixes_or_rewrites",
    "should_retry": boolean
}}

{parser.get_format_instructions()}
"""

            
            response = self.llm.invoke(validation_prompt)
            validation_result = parser.parse(response.content)
            
            state["validation_result"] = validation_result
            state["validation_attempts"] = validation_attempts + 1
            
            logger.info(f"Validation result: accuracy={validation_result.accuracy_score:.2f}, correct={validation_result.is_correct}")
            
            return state
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            fallback_validation = ValidationResult(
                is_correct=True,
                accuracy_score=0.8,
                feedback=f"Validation process failed: {e}",
                improvement_suggestions="Manual review recommended",
                should_retry=False
            )
            state["validation_result"] = fallback_validation
            state["validation_attempts"] = state.get("validation_attempts", 0) + 1
            return state

    def enhance_query_with_feedback(self, state: SQLState) -> SQLState:

        try:
            original_query = state.get("query", "")
            validation_result = state.get("validation_result")
            current_sql = state.get("generated_query", "")
            current_result = state.get("result", "")
            
            parser = PydanticOutputParser(pydantic_object=QueryEnhancement)
            
            enhancement_prompt = f"""
You are a query enhancement expert for SQL database operations.

ORIGINAL QUERY: "{original_query}"
CURRENT GENERATED SQL: {current_sql}
CURRENT RESULT: {current_result}
VALIDATION FEEDBACK:
- Score: {validation_result.accuracy_score}
- Issues: {validation_result.feedback}
- Suggestions: {validation_result.improvement_suggestions}

DATABASE CONTEXT: {self.database_info if self.database_info else "No database info available"}

TASK: Add System Validation Note to original query with specific fixes.

Example:
Original: "Show top customers by orders"
Issues: "Missing ORDER BY, should use customer_name not name"
Output: "Show top customers by orders.

System Validation Note: Use 'customer_name' column, add ORDER BY total_orders DESC, include LIMIT for top results."

REQUIRED JSON (no extra text):
{{
    "enhanced_query": "original_query + system_validation_note",
    "enhancement_reasoning": "brief_changes_explanation", 
    "confidence": 0.0-1.0
}}

{parser.get_format_instructions()}
"""
            
            response = self.llm.invoke(enhancement_prompt)
            enhancement = parser.parse(response.content)
            
            state["query"] = enhancement.enhanced_query
            logger.info(f"Query enhanced: {enhancement.enhanced_query}")

            state["validation_result"] = None
            state["generated_query"] = None
            state["execution_result"] = None
            state["result"] = None
            
            return state
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            fallback_query = state["query"] + "\nSystem Validation Note: " + state["validation_result"].feedback
            state["query"] = fallback_query
            logger.info(f"Query enhanced(fallback): {fallback_query}")
            state["validation_result"] = None
            state["generated_query"] = None
            state["execution_result"] = None
            state["result"] = None

            return state

    def error_handling(self, state: SQLState) -> SQLState:
        error = state.get("error", "Unknown error occurred")
        retry_count = state.get("retry_count", 0)
        
        logger.error(f"Error in SQL workflow (retry {retry_count}/{self.max_retries})")
        
        if retry_count >= self.max_retries:
            logger.error(f"Maximum retry limit ({self.max_retries}) exceeded")
            return self._simple_error_handling(state)
        
        try:
            return self._llm_error_correction(state)
        except Exception as e:
            logger.error(f"LLM error correction failed: {e}")
            return self._simple_error_handling(state)

    def _llm_error_correction(self, state: SQLState) -> SQLState:
        error = state.get("error", "Unknown error occurred")
        query = state.get("query", "")
        sql_query = state.get("generated_query", "")
        current_node = self._get_current_node_context(state)
        
        parser = PydanticOutputParser(pydantic_object=ErrorCorrection)
        
        error_handling_prompt = f"""
You are an error handler for SQL workflows.

CURRENT ERROR:
- Failed Node: {current_node}
- User Query: "{query}"
- Generated SQL: "{sql_query}"
- Error: "{error}"
- Retry: {state.get("retry_count", 0)}/{self.max_retries}

DATABASE INFO: {self.database_info if self.database_info else "Not loaded"}

RULES:
1. Database/schema errors → next_node: "get_database_info"
2. SQL syntax/logic errors → Fix and next_node: "generate_sql_query"  
3. Too many retries → next_node: "END"

OUTPUT FORMAT:
Original query + System Error Note with fix instructions.

Example:
Original: "Show top customers"
Error: Table 'customer' doesn't exist
Output: "Show top customers.

System Error Note: Use 'customers' table (plural) instead of 'customer'. Check table names in schema."

REQUIRED JSON (no extra text):
{{
    "next_node": "get_database_info|generate_sql_query|END",
    "corrected_user_query": "original_query + system_error_note",
    "reasoning": "brief_fix_explanation",
    "confidence": 0.0-1.0
}}

{parser.get_format_instructions()}
"""
        
        response = self.llm.invoke(error_handling_prompt)
        correction = parser.parse(response.content)
        
        valid_nodes = ["get_database_info", "generate_sql_query", "END"]
        if correction.next_node not in valid_nodes:
            logger.warning(f"Invalid next_node '{correction.next_node}', defaulting to END")
            correction.next_node = "END"
        
        state["retry_count"] = state.get("retry_count", 0) + 1
        
        if correction.corrected_user_query and correction.corrected_user_query != query:
            state["query"] = correction.corrected_user_query
            logger.info(f"Query enhanced with error context: {correction.corrected_user_query}")
        
        if correction.next_node == "generate_sql_query":
            state["generated_query"] = None
            logger.info("Cleared generated_query to force regeneration with enhanced query")
        
        state["error"] = None
        state["next_error_node"] = correction.next_node
        
        logger.info(f"Error correction applied.")
        logger.info(f"Next node to execute: {correction.next_node}")
        return state

    def _get_current_node_context(self, state: SQLState) -> str:
        if not state.get("database_info_loaded"):
            return "get_database_info"
        elif not state.get("generated_query"):
            return "generate_sql_query" 
        elif not state.get("execution_result"):
            return "execute_sql_query"
        else:
            return "unknown"

    def _simple_error_handling(self, state: SQLState) -> SQLState:
        """Fallback simple error handling"""
        error = state.get("error", "Unknown error occurred")
        
        if "SQL execution failed" in error:
            state["result"] = f"Database query execution failed.\n\nDetails: {error}"
        elif "SQL generation failed" in error:
            state["result"] = f"Could not generate SQL query.\n\nDetails: {error}"
        elif "Database info retrieval failed" in error:
            state["result"] = f"Could not connect to database.\n\nDetails: {error}"
        elif "not suitable for database operations" in error:
            state["result"] = f"This query is not related to database operations.\n\nPlease ask database-related questions."
        else:
            state["result"] = f"An error occurred: {error}"
        
        state["next_error_node"] = "END"
        return state

    def _route_condition(self, state: SQLState) -> Literal["enhance_query_with_feedback","validate_result","get_database_info", "generate_sql_query", "execute_sql_query", "error_handling", "END"]:
        route = state.get("route_decision")

        if state.get("error"):
            logger.info(f"-->error_handling")            
            return "error_handling"
        
        if not route:
            logger.info(f"-->error_handling")
            return "error_handling"
        
        if route.workflow_type == WorkflowType.INVALID:
            logger.info(f"-->END")
            return "END"
        
        if not state.get("database_info_loaded") and not state.get("generated_query") and not state.get("execution_result"):
            if route.workflow_type == WorkflowType.DIRECT_EXECUTE:
                logger.info(f"START-->execute_sql_query")
                return "execute_sql_query"
            else:
                logger.info(f"START-->get_database_info")
                return "get_database_info"
        
        if state.get("database_info_loaded") and not state.get("generated_query") and not state.get("execution_result"):
            if route.workflow_type == WorkflowType.INFO_ONLY:
                logger.info(f"get_database_info-->END")
                state["result"] = state["database_info"]
                return "END"
            else:
                logger.info(f"get_database_info-->generate_sql_query")
                return "generate_sql_query"
        
        if state.get("generated_query") and not state.get("execution_result"):
            if route.workflow_type == WorkflowType.GENERATE_QUERY_ONLY:
                logger.info(f"generate_sql_query-->validate_result")
                state["result"] = state["generated_query"]
                return "validate_result"
            else:
                logger.info(f"generate_sql_query-->execute_sql_query")
                return "execute_sql_query"
            
        if state.get("execution_result") and not state.get("validation_result"):
            state["result"] = state["execution_result"]
            return "validate_result"
        
        if state.get("validation_attempts") >= self.max_validation_attempts:
            return "END"
        
        if state.get("validation_result"):
            if state["validation_result"].accuracy_score >= 0.8:
                return "END"
            else:
                return "enhance_query_with_feedback"
        
        return "END"

    def _error_route_condition(self, state: SQLState) -> Literal["get_database_info", "generate_sql_query", "execute_sql_query", "END"]:
        next_node = state.get("next_error_node", "END")
        logger.info(f"Error handling routing to: {next_node}")
        return next_node
            
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(SQLState)
        workflow.add_node("llm_routing", self._llm_routing_node)
        workflow.add_node("get_database_info", self.get_database_info)
        workflow.add_node("generate_sql_query", self.generate_sql_query)
        workflow.add_node("execute_sql_query", self.execute_sql_query)
        workflow.add_node("error_handling", self.error_handling)
        workflow.add_node("validate_result", self.validate_result)
        workflow.add_node("enhance_query_with_feedback", self.enhance_query_with_feedback)

        workflow.set_entry_point("llm_routing")
        workflow.add_conditional_edges(
            "llm_routing", 
            self._route_condition, {
                "get_database_info": "get_database_info",
                "execute_sql_query": "execute_sql_query",
                "END": END
            })

        workflow.add_conditional_edges(
            "get_database_info",
            self._route_condition, {
                "END": END,
                "generate_sql_query": "generate_sql_query",
                "error_handling": "error_handling"
            })

        workflow.add_conditional_edges(
            "generate_sql_query",
            self._route_condition, {
                "validate_result": "validate_result",
                "execute_sql_query": "execute_sql_query",
                "error_handling": "error_handling"
            })

        workflow.add_conditional_edges(
            "execute_sql_query",
            self._route_condition, {
                "validate_result": "validate_result",
                "error_handling": "error_handling"
            })

        workflow.add_conditional_edges(
            "validate_result",
            self._route_condition,{
                "END":END,
                "enhance_query_with_feedback":"enhance_query_with_feedback"
            }
        )

        workflow.add_edge("enhance_query_with_feedback","generate_sql_query")

        workflow.add_conditional_edges(
            "error_handling",
            self._error_route_condition, {
                "END": END,
                "get_database_info": "get_database_info",
                "generate_sql_query": "generate_sql_query", 
                "execute_sql_query": "execute_sql_query"
            })
        
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def process(self, query: str = None) -> AgentResult:
        if query:
            self.query = query
            
        logger.info(f"Processing SQL query: {self.query}")

        try:
            config = {"configurable": {"thread_id": "sql_agent_thread"}}
            initial_state = {"query": self.query}
            final_state = self.app.invoke(initial_state, config)
            
            if final_state.get("error"):
                return AgentResult(
                    success=False,
                    error=final_state["error"],
                    data=None,
                    metadata={
                        "agent": self.agent_name,
                        "query": self.query
                    }
                )
            
            result_data = {
                "output": final_state.get("result", "No result available"),
                "query": self.query,
                "generated_sql": final_state.get("generated_query"),
                "execution_result": final_state.get("execution_result"),
                "database_info": final_state.get("database_info")
            }

            result_df = None
            if self.the_answer:
                result_df = self.the_answer.dataframe
            
            logger.info("SQL agent processing completed successfully")
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent": self.agent_name,
                    "query": self.query,
                    "db_tables": self.db_tables,
                    "doc_path": self.doc_path,
                    "columnInfo_path": self.columnInfo_path,
                    "dataframe": result_df,
                }
            )
            
        except Exception as e:
            error_msg = f"SQL agent processing failed: {str(e)}"
            logger.error(error_msg)
            return AgentResult(
                success=False,
                error=error_msg,
                data=None,
                metadata={
                    "agent": self.agent_name,
                    "query": self.query
                }
            )


def example_usage():
    """Example usage with corrected parameters"""
    import os
    from utils.db_dao import DatabaseManager
    
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
            print("GROQ_API_KEY not found")
            return
        
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
        db_manager = DatabaseManager(db_params)
        
        agent = SQLQuerryAgent(
            llm=llm,
            db_manager=db_manager,
            doc_path="columns.pdf"
        )
        
        test_queries = [
            "get me top 5 customers by revenue"
        ]
        
        for query in test_queries:
            print(f"\nTesting: {query}")
            result = agent.process(query)
            if result.success:
                print("Generated SQL:\n", result.data["generated_sql"])
                if result.data["execution_result"]:
                    print("Execution Result:\n", pd.DataFrame(json.loads(result.data["execution_result"])))
                else:
                    print("Execution Result:\n",result.data["execution_result"])
            else:
                print("Error:", result.error)
            
    except Exception as e:
        print(f"Example failed: {e}")

if __name__ == "__main__":
    example_usage()