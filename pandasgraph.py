import os
from typing import Any, Dict, Literal, Optional, TypedDict
from langchain_groq import ChatGroq
from utils.loggerCenter import LoggerCenter
from utils.base_agent import to_dataframe_safe,BaseSpecializedAgent,CodeOutput,AgentResult,clean_imports_from_code,is_safe_code
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import pandas as pd
from utils.vectordeneme import ContextFind  
from utils.document import DocumentProcessor
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from enum import Enum
import json

logger = LoggerCenter().get_logger()



#############################################################################
class ErrorCorrection(BaseModel):
    next_node: str = Field(description="Next node to execute: 'get_data_info', 'generate_pandas_code', or 'END'")
    corrected_user_query: Optional[str] = Field(description="Enhanced user query that includes error context and fix instructions. Always provided for code generation errors.")
    reasoning: str = Field(description="Explanation of error analysis and query enhancement strategy")
    confidence: float = Field(description="Confidence level between 0.0 and 1.0")
#############################################################################

#################################################################################
class ValidationResult(BaseModel):
    is_correct: bool = Field(description="Whether the result correctly answers the user query")
    accuracy_score: float = Field(ge=0.0, le=1.0, description="Accuracy score of the result (0.0 to 1.0)")
    feedback: str = Field(description="Detailed feedback about what's wrong or missing")
    improvement_suggestions: str = Field(description="Specific suggestions to improve the query/code")
    should_retry: bool = Field(description="Whether the query should be retried with improvements")

class QueryEnhancement(BaseModel):
    enhanced_query: str = Field(description="Enhanced query with improvement notes and specific guidance")
    enhancement_reasoning: str = Field(description="Explanation of what was enhanced and why")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the enhancement")
#############################################################################

class PythonExecutionResult(BaseModel):
    success: bool
    code: str
    dataframe: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None
    formatted_output: str = ""

    class Config:
        arbitrary_types_allowed = True

class WorkflowType(str, Enum):
    DATA_INFO_ONLY = "data_info_only"
    GENERATE_CODE_ONLY = "generate_code_only"
    EXECUTE_CODE = "execute_code"
    DIRECT_EXECUTE = "direct_execute"
    INVALID = "invalid"


class DataRoutingDecision(BaseModel):
    workflow_type: WorkflowType = Field(description="The type of workflow to execute")
    reasoning: str = Field(description="Explanation of the routing decision")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in routing decision")

class DataAnalysisState(TypedDict):
    query: str
    route_decision: Optional[DataRoutingDecision]
    routing_reasoning: Optional[str]
    confidence_score: Optional[float]
    result: Optional[str]
    error: Optional[str]
    metadata: Optional[Dict[str, Any]]
    data_info_loaded: Optional[bool]
    data_info: Optional[str]
    generated_code: Optional[str]
    execution_result: Optional[str]
    next_error_node:Optional[str]
    retry_count:Optional[int]
    validation_attempts: Optional[int]  # Track validation attempts
    validation_result: Optional[ValidationResult]
    is_validated: Optional[bool]
    final_result: Optional[str]

class DataAnalysisAgent(BaseSpecializedAgent):

    def __init__(self, llm: ChatGroq, df: pd.DataFrame, doc_path: str = None, column_info_path: str = None):
        super().__init__("DataAnalysisAgent", llm)

        self.df = df
        self.doc_path = doc_path
        self.column_info_path = column_info_path
        self.max_retries = 5
        self.max_validation_attempts = 4
        
        self.context_finder = None
        if doc_path:
            try:
                self.context_finder = ContextFind(doc_path)
                logger.info(f"Context finder initialized for: {doc_path}")
            except Exception as e:
                logger.error(f"Failed to initialize context finder: {e}")
                self.context_finder = None

        self.doc_process = None
        self.column_info = None
        if column_info_path:
            try:
                self.doc_process = DocumentProcessor()
                self.column_info = self.doc_process.extract_text_from_documents(column_info_path)
                logger.info(f"Column info loaded from: {column_info_path}")
            except Exception as e:
                logger.error(f"Failed to load column info: {e}")
                self.column_info = None

        self.current_context = "No data available"
        self.the_answer = None
        self.data_info = None
        self.app = self._build_graph()

    def _llm_routing_node(self, state: DataAnalysisState) -> DataAnalysisState:
        query = state["query"]
        logger.info(f"Making intelligent LLM-based routing decision for: {query}")
        
        parser = PydanticOutputParser(pydantic_object=DataRoutingDecision)
        
        routing_prompt = f"""
You are an intelligent routing system for data analysis tasks.
Your job is to analyze the USER QUERY and decide which data analysis operations should be executed.

USER QUERY:
"{query}"

AVAILABLE TOOLS/AGENTS:
1. get_data_info     →  Retrieves DataFrame schema, statistics, context, and column information
2. generate_pandas_code → Generates pandas code from user requirements and data context
3. execute_python_code  → Executes pandas code and returns structured results

ROUTING RULES:
1. If the user asks for data details/summary/info → classify as "data_info_only"
2. If the user only wants pandas code → classify as "generate_code_only"
3. If the user wants both code generation and execution → classify as "execute_code"
4. If the user provides pandas code and asks to run it → classify as "direct_execute"
5. If the user query is not related to data analysis → classify as "invalid"

WORKFLOW:
- "data_info_only": get_data_info → END
- "generate_code_only": get_data_info → generate_pandas_code → END
- "execute_code": get_data_info → generate_pandas_code → execute_python_code → END
- "direct_execute": execute_python_code → END
- "invalid": → END

OUTPUT FORMAT:
You must output only valid JSON matching the schema below.
Do not include explanations, markdown, or code blocks.
Output JSON only.

EXAMPLE (user asks for top 10 customers analysis):
{{
    "workflow_type": "execute_code",
    "reasoning": "User wants data analysis results, need to load data info, generate code, and execute",
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
                    "error": "Routing classification failed: Query is not suitable for data analysis operations."
                }
            
            return {
                **state,
                "route_decision": routing_decision,
                "routing_reasoning": routing_decision.reasoning,
                "confidence_score": routing_decision.confidence
            }
            
        except Exception as e:
            logger.error(f"LLM routing error: {e}")
            fallback_decision = DataRoutingDecision(
                workflow_type=WorkflowType.INVALID,
                reasoning=f"Routing classification failed: Error: {e}",
                confidence=0.8
            )
            
            return {
                **state,
                "route_decision": fallback_decision,
                "error": f"Routing classification failed: Error: {e}"
            }

    def get_data_info(self, state: DataAnalysisState) -> DataAnalysisState:
        try:
            logger.info(f"Getting the data info.")
            if self.df is None:
                state["error"] = "No DataFrame available for analysis"
                return state

            context_info = "No context available"
            column_info_text = "No column info available"
            doc_link = "Not provided"
            column_info_link = "Not provided"

            num_rows = self.df.shape[0]
            num_cols = self.df.shape[1]
            columns_list = [str(col).replace('{', '[').replace('}', ']') for col in self.df.columns]
            columns = ", ".join(columns_list)
            d_types = dict(self.df.dtypes)

            sample_df = self.df.head(3)
            sample_str = sample_df.to_string(index=False)
            safe_sample = sample_str.replace('{', '[').replace('}', ']')

            if self.context_finder and state["query"]:
                try:
                    context_info = self.context_finder.return_context(state["query"], top_k=3)
                    if not context_info.strip():
                        context_info = "No relevant context found"
                    doc_link = self.doc_path
                    self.current_context = context_info
                except Exception as e:
                    logger.error(f"Error getting context: {e}")

            if self.column_info:
                try:
                    column_info_text = self.column_info
                    column_info_link = self.column_info_path
                except Exception as e:
                    logger.error(f"Error getting column info: {e}")

            self.data_info = f"""
DATA ANALYSIS CONTEXT:
Shape: {num_rows} rows x {num_cols} columns
Columns: {columns}
Data Types: {d_types}

Sample Data (first 3 rows):
{safe_sample}

RELEVANT CONTEXT:
{context_info}

COLUMN INFORMATION:
{column_info_text}

Context retrieved from: {doc_link}
Column Info retrieved from: {column_info_link}
"""
            logger.info(f"Data Info uploaded: {self.data_info}")

            state["data_info_loaded"] = True
            state["data_info"] = self.data_info
            return state

        except Exception as e:
            state["error"] = f"Data info retrieval failed: {e}"
            logger.error(state["error"])
            return state

    def generate_pandas_code(self, state: DataAnalysisState) -> DataAnalysisState:
        try:
            query=state["query"]
            logger.info(f"Code is being generated. User query ->{query}")

            if not state["data_info_loaded"]:
                logger.info("Data info not loaded, loading automatically...")
                self.get_data_info(state)
            
            parser=PydanticOutputParser(pydantic_object=CodeOutput)
            pandas_prompt = f"""
You are an expert data analyst. Generate clean pandas code that answers the request using the loaded data information.

=== LOADED DATA INFORMATION ===
{self.data_info}

=== INSTRUCTIONS ===
1. Use the DataFrame schema and context information provided above
2. Pay special attention to the business context to understand the requirements
3. Do NOT include anything outside of CODE and an Explanation
4. Your response must be EXACTLY one valid JSON object, no explanations, no markdown, no extra text

CRITICAL JSON OUTPUT FORMAT:
{{
  "code": "your_pandas_code_here",
}}

{parser.get_format_instructions}

CRITICAL REQUIREMENTS:
1. Generate ONLY pandas code without markdown formatting
2. Use proper column names from the loaded schema above
3. The DataFrame is already available as 'df'
4. Always assign final results to a variable named 'result'
5. Use only pandas operations - pandas is imported as 'pd'
6. Do NOT include any import statements
7. Ensure the code is safe (no file operations, system calls, etc.)

User Request: {query}
"""
            
            raw_response = self.llm.invoke(pandas_prompt)
            response = parser.parse(raw_response.content)
           
            try:
                if isinstance(response, dict):
                    if "text" in response:
                        pandas_code = response["text"].code if hasattr(response["text"], 'code') else str(response["text"])
                    else:
                        pandas_code = response.get("code", str(response))
                else:
                    pandas_code = response.code if hasattr(response, 'code') else str(response)
                
               
                logger.info(f"Pandas code generated: {pandas_code}")
                state["generated_code"] = pandas_code
                return state
                
            except Exception as e:
                state["error"] = f"Pandas code generation failed: Error while cleaning response {e}"
                logger.error(state["error"])
                return state
                                
        except Exception as e:
            state["error"] = f"Pandas code generation failed: Error while producing response {e}"
            logger.error(state["error"])
            return state

    def execute_python_code(self, state: DataAnalysisState) -> DataAnalysisState:
        
        if not state["generated_code"] and state["route_decision"].workflow_type != WorkflowType.DIRECT_EXECUTE:
            logger.info("Code not generated, generating automatically...")
            self.generate_pandas_code(state)
        
        if state["route_decision"].workflow_type == WorkflowType.DIRECT_EXECUTE:
            state["generated_code"] = state["query"]
        

        try:
            pandas_code = state["generated_code"]
            logger.info(f"Executing the Code.")

            if self.df is None:
                state["error"] = "No DataFrame available for code execution"
                return state
            
            if not is_safe_code(pandas_code):
                state["error"] = "Code contains potentially dangerous operations and cannot be executed"
                logger.error(state["error"])
                return state

            safe_globals = {
                'pd': pd,
                'df': self.df.copy(),
                'np': pd.np if hasattr(pd, 'np') else None,
                '__builtins__': {
                    'len': len, 'str': str, 'int': int, 'float': float,
                    'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                    'range': range, 'enumerate': enumerate, 'zip': zip,
                    'sorted': sorted, 'reversed': reversed, 'sum': sum,
                    'min': min, 'max': max, 'abs': abs, 'round': round,
                    'print': print,"__import__":__import__
                }
            }
            #pandas_code=clean_imports_from_code(pandas_code)
            local_vars = {}
            exec(pandas_code, safe_globals, local_vars)
            
            result = local_vars.get("result", None)
            
            if result is not None:

                if not isinstance(result, pd.DataFrame):
                    result = to_dataframe_safe(result)
                
                if isinstance(result, pd.DataFrame):
                    if result.empty:
                        result_text = f"""Code Executed Successfully!

Code: {pandas_code}

Result: Empty DataFrame"""
                        
                        state["execution_result"] = result.to_json(orient='records')
                        self.the_answer = PythonExecutionResult(
                            success=True,
                            code=pandas_code,
                            dataframe=result,
                            formatted_output=result_text
                        )
                        return state

                    num_rows = len(result)
                    if num_rows > 20:
                        display_result = result.head(20)
                        result_text = f"""Code Executed Successfully!

Code: {pandas_code}

Result (showing first 20 rows of {num_rows}):
{display_result.to_string(index=False)}
"""
                    else:
                        result_text = f"""Code Executed Successfully!

Code: {pandas_code}

Result ({num_rows} rows):
{result.to_string(index=False)}"""
                    

                    state["execution_result"] = result.to_json(orient='records')
                    self.the_answer = PythonExecutionResult(
                        success=True,
                        code=pandas_code,
                        dataframe=result,
                        formatted_output=result_text
                    )

                    logger.info(f"Code executed successfully, returned {result.head()}")
                    return state
                
                else:
                    result_text = f"""Code Executed Successfully!

Code: {pandas_code}

Result: {result}"""
                    
                    state["execution_result"] = str(result)
                    self.the_answer = PythonExecutionResult(
                        success=True,
                        code=pandas_code,
                        dataframe=None,
                        formatted_output=result_text
                    )
                    logger.info(f"Code executed successfully.")
                    return state
            else:
                result_text = f"""Code Executed Successfully!

Code: {pandas_code}

Result: Code executed but no 'result' variable was assigned"""
                
                state["execution_result"] = result_text
                logger.info(f"Code executed successfully, returned no result ")
                return state

        except Exception as e:
            import traceback
            error_msg = f"Code execution failed: {str(e)}\nTraceback:\n{traceback.format_exc()}"
            state["error"] = error_msg
            logger.error(state["error"])
            return state

#######################################################################################

    def validate_result(self, state: DataAnalysisState) -> DataAnalysisState:
        
        try:
            original_query = state.get("query", "")
            generated_code = state.get("generated_code", "")
            execution_result = state.get("execution_result", "")
            validation_attempts = state.get("validation_attempts", 0)
            
            logger.info(f"Validating result (attempt {validation_attempts + 1}/{self.max_validation_attempts})")

            if validation_attempts >=self.max_validation_attempts:
                return state
            
            parser = PydanticOutputParser(pydantic_object=ValidationResult)
            
            validation_prompt = f"""
You are an expert data analysis validator. Your task is to evaluate whether the generated result correctly and completely answers the user's original query.

=== ORIGINAL USER QUERY ===
"{original_query}"

=== GENERATED CODE ===
{generated_code}

=== EXECUTION RESULT ===
{execution_result}

=== DATA CONTEXT ===
{self.data_info if self.data_info else "No data info available"}

=== VALIDATION CRITERIA ===
1. Does the result directly answer what the user asked for?
2. Is the data analysis approach correct and logical?
3. Are the column names and operations appropriate?
4. Is the result format suitable for the query type?
5. Are there any logical errors or missing elements?

=== ACCURACY SCORING GUIDE ===
- 1.0 (Perfect): Result completely and accurately answers the query
- 0.8-0.9 (Good): Result answers the query with minor issues or could be enhanced
- 0.6-0.7 (Partial): Result partially answers but has significant gaps
- 0.4-0.5 (Poor): Result has major issues but shows some relevance
- 0.0-0.3 (Wrong): Result doesn't answer the query or is completely wrong

=== CRITICAL RULES ===
- If accuracy_score >= 0.8, set is_correct = true and should_retry = false
- If accuracy_score < 0.8, set is_correct = false and should_retry = true
- Provide specific, actionable feedback and improvement suggestions
- Focus on what's missing or wrong, not just what's good

REQUIRED JSON OUTPUT (no additional text):
{{
    "is_correct": boolean,
    "accuracy_score": float_between_0_and_1,
    "feedback": "detailed_feedback_about_issues",
    "improvement_suggestions": "specific_suggestions_to_fix_issues",
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

    def enhance_query_with_feedback(self, state: DataAnalysisState) -> DataAnalysisState:

        try:
            original_query = state.get("query", "")
            validation_result = state.get("validation_result")
            current_code = state.get("generated_code", "")
            current_result = state.get("result", "")
            
            parser = PydanticOutputParser(pydantic_object=QueryEnhancement)
            
            enhancement_prompt = f"""
You are a query enhancement expert. Your task is to improve the user query by incorporating validation feedback and specific improvement suggestions.

=== ORIGINAL USER QUERY ===
"{original_query}"

=== CURRENT GENERATED CODE ===
{current_code}

=== CURRENT RESULT ===
{current_result}

=== VALIDATION FEEDBACK ===
Accuracy Score: {validation_result.accuracy_score}
Issues Found: {validation_result.feedback}
Suggestions: {validation_result.improvement_suggestions}

=== DATA CONTEXT ===
{self.data_info if self.data_info else "No data info available"}

=== ENHANCEMENT TASK ===
Create an enhanced query that:
1. Preserves the original user intent
2. Incorporates the validation feedback
3. Includes specific technical guidance to fix identified issues
4. Provides clear instructions for correct column usage, operations, filtering, etc.
5. Addresses any logical gaps or missing elements

=== ENHANCEMENT EXAMPLES ===
Original: "Show top 5 players"
Issues: "Column 'player_name' not found, should use 'scorer'"
Enhanced: "Show top 5 players by goals scored. IMPORTANT: Use 'scorer' column for player names (not 'player_name'), group by 'scorer' and sum 'goals', then sort by total goals descending and take top 5."

Original: "Calculate average goals"
Issues: "Not clear if per player or overall, result format unclear"
Enhanced: "Calculate average goals per player and display results clearly. GROUP BY 'scorer' column, calculate mean of 'goals' column, round to 2 decimal places, and sort by average descending."

REQUIRED JSON OUTPUT (no additional text):
{{
    "enhanced_query": "enhanced_query_with_specific_guidance",
    "enhancement_reasoning": "explanation_of_changes_made",
    "confidence": float_between_0_and_1
}}

{parser.get_format_instructions()}
"""
            
            response = self.llm.invoke(enhancement_prompt)
            enhancement = parser.parse(response.content)
            
            state["query"] = enhancement.enhanced_query
            logger.info(f"Query enhanced: {enhancement.enhanced_query}")

            state["validation_result"]=None
            state["generated_code"] = None
            state["execution_result"] = None
            state["result"] = None
            
            
            return state
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            fallback_query=state["query"]+"\nNotes: "+state["validation_result"].feedback
            state["query"]=fallback_query
            logger.info(f"Query enhanced(fallback): {fallback_query}")
            state["validation_result"] = None
            state["generated_code"] = None
            state["execution_result"] = None
            state["result"] = None

            return state
    
    #######################################################################
    
    def error_handling(self, state: DataAnalysisState) -> DataAnalysisState:
        error = state.get("error", "Unknown error occurred")
        retry_count = state.get("retry_count", 0)
        
        logger.error(f"Error in data analysis workflow (retry {retry_count}/{self.max_retries})")
        
        if retry_count >= self.max_retries:
            logger.error(f"Maximum retry limit ({self.max_retries}) exceeded")
            return self._simple_error_handling(state)
        
        try:
            return self._llm_error_correction(state)
        except Exception as e:
            logger.error(f"LLM error correction failed: {e}")
            return self._simple_error_handling(state)

    def _llm_error_correction(self, state: DataAnalysisState) -> DataAnalysisState:
        error = state.get("error", "Unknown error occurred")
        query = state.get("query", "")
        pandas_code = state.get("generated_code", "")
        current_node = self._get_current_node_context(state)
        
        parser = PydanticOutputParser(pydantic_object=ErrorCorrection)
        
        error_handling_prompt = f"""
    You are an error handling agent for data analysis workflows. Your task is to analyze the error, determine the root cause, and embed the error context into a corrected query for code regeneration.

    CURRENT ERROR CONTEXT:
    - CURRENT NODE FAILED: {current_node}
    - USER QUERY: "{query}"
    - GENERATED CODE: "{pandas_code}"
    - ERROR MESSAGE: "{error}"
    - RETRY COUNT: {state.get("retry_count", 0)}/{self.max_retries}

    AVAILABLE DATA INFO: 
    {self.data_info if self.data_info else "Data info not loaded yet"}

    AVAILABLE WORKFLOW NODES:
    1. get_data_info
    2. generate_pandas_code
    3. execute_python_code
    4. END

    ERROR ANALYSIS RULES:
    1. If data loading/schema loading failed → next_node: "get_data_info"
    2. If code generation failed due to missing context → next_node: "get_data_info"
    3. For pandas syntax/logic errors → Analyze the error and embed the fix instructions into corrected_user_query → next_node: "generate_pandas_code"
    4. For execution errors → Analyze what went wrong and create an enhanced query with error context → next_node: "generate_pandas_code"
    5. If error is unfixable or too many retries → next_node: "END"

    QUERY ENHANCEMENT STRATEGY:
    When creating corrected_user_query, include:
    - Original user intent
    - Specific error context and what went wrong
    - Technical hints about correct approach
    - Data structure guidance if relevant

    CORRECTION TASKS (IMPORTANT):
    - corrected_user_query → ALWAYS provide an enhanced query that includes error analysis and fix instructions

    Example Enhanced Queries:
    Original: "Show top 5 players"
    Error: "KeyError: 'player_name'"
    Enhanced: "Show top 5 players by goals scored. Note: Use 'scorer' column instead of 'player_name' as the column name. The available columns should be checked first."

    Original: "Calculate average goals"
    Error: "TypeError: cannot perform mean on string column"
    Enhanced: "Calculate average goals per player. Important: Convert the goals column to numeric first using pd.to_numeric() and handle any non-numeric values appropriately."

    !!!CRITICAL OUTPUT INSTRUCTIONS!!!
    - You MUST return ONLY a single **valid JSON object**.
    - DO NOT include any text, markdown, or explanation outside the JSON.
    - All string values MUST be properly JSON-escaped.
    - Confidence MUST be a float between 0.0 and 1.0.
    - JSON MUST start with '{' and end with '}'.

    Required JSON format:
    {{
    "next_node": "get_data_info | generate_pandas_code | END",
    "corrected_user_query": "string with enhanced query including error context and fix instructions",
    "reasoning": "string explaining the error analysis and query enhancement strategy",
    "confidence": float
    }}

    Example Response:
    {{
    "next_node": "generate_pandas_code",
    "corrected_user_query": "Show top 5 players by total goals scored. Important: Use 'scorer' column for player names (not 'player_name'), and sum up goals per player using groupby('scorer')['goals'].sum(). Handle any missing values appropriately.",
    "reasoning": "The error occurred because the code tried to access 'player_name' column which doesn't exist. The correct column is 'scorer'. Enhanced the query to include this specific guidance and the correct groupby approach.",
    "confidence": 0.85
    }}

    {parser.get_format_instructions()}
    """

        response = self.llm.invoke(error_handling_prompt)
        correction = parser.parse(response.content)
        
        valid_nodes = ["get_data_info", "generate_pandas_code", "END"]
        if correction.next_node not in valid_nodes:
            logger.warning(f"Invalid next_node '{correction.next_node}', defaulting to END")
            correction.next_node = "END"
        
        state["retry_count"] = state.get("retry_count", 0) + 1
        
        if correction.corrected_user_query and correction.corrected_user_query != query:
            state["query"] = correction.corrected_user_query
            logger.info(f"Query enhanced with error context: {correction.corrected_user_query}")
        
        if correction.next_node == "generate_pandas_code":
            state["generated_code"] = None
            logger.info("Cleared generated_code to force regeneration with enhanced query")
        
        state["error"] = None
        state["next_error_node"] = correction.next_node
        
        logger.info(f"Error correction applied.")
        logger.info(f"Next node to execute: {correction.next_node}")
        return state

    def _simple_error_handling(self, state: DataAnalysisState) -> DataAnalysisState:
        """Fallback simple error handling"""
        error = state.get("error", "Unknown error occurred")
        
        if "Code execution failed" in error:
            state["result"] = f"Pandas code execution failed.\n\nDetails: {error}"
        elif "generation failed" in error:
            state["result"] = f"Could not generate pandas code.\n\nDetails: {error}"
        elif "Data info retrieval failed" in error:
            state["result"] = f"Could not load data information.\n\nDetails: {error}"
        elif "not suitable for data analysis" in error:
            state["result"] = f"This query is not related to data analysis operations.\n\nPlease ask data analysis questions."
        else:
            state["result"] = f"An error occurred: {error}"
        
        state["next_error_node"] = "END"
        return state

    def _error_route_condition(self, state: DataAnalysisState) -> Literal["get_data_info", "generate_pandas_code", "execute_python_code", "END"]:
        next_node = state.get("next_error_node", "END")
        logger.info(f"Error handling routing to: {next_node}")
        return next_node
    
    #######################################################################
    def _get_current_node_context(self, state: DataAnalysisState) -> str:
        if not state.get("data_info_loaded"):
            return "get_data_info"
        elif not state.get("generated_code"):
            return "generate_pandas_code" 
        elif not state.get("execution_result"):
            return "execute_python_code"
        else:
            return "unknown"
    
    def _route_condition(self, state: DataAnalysisState) -> Literal["enhance_query_with_feedback","validate_result","get_data_info", "generate_pandas_code", "execute_python_code", "error_handling", "END"]:
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
        
        if not state.get("data_info_loaded") and not state.get("generated_code") and not state.get("execution_result"):
            if route.workflow_type == WorkflowType.DIRECT_EXECUTE:
                logger.info(f"START-->execute_python_code")
                return "execute_python_code"
            else:
                logger.info(f"START-->get_data_info")
                return "get_data_info"
        
        if state.get("data_info_loaded") and not state.get("generated_code") and not state.get("execution_result"):
            if route.workflow_type == WorkflowType.DATA_INFO_ONLY:
                logger.info(f"get_data_info-->END")
                state["result"]=state["data_info"]
                return "END"
            else:
                logger.info(f"get_data_info-->generate_pandas_code")
                return "generate_pandas_code"
        
        if state.get("generated_code") and not state.get("execution_result"):
            if route.workflow_type == WorkflowType.GENERATE_CODE_ONLY:
                logger.info(f"generate_pandas_code-->validate_result")
                state["result"]=state["generated_code"]
                return "validate_result"
            else:
                logger.info(f"generate_pandas_code-->execute_python_code")
                return "execute_python_code"
            
        if state.get("execution_result") and not state.get("validation_result"):
            state["result"]=state["execution_result"]
            return "validate_result"
        
        if state.get("validation_attempts")>=self.max_validation_attempts:
            return "END"
        
        if state.get("validation_result"):
            if state["validation_result"].accuracy_score >= 0.8:
                return "END"
            else:
                return "enhance_query_with_feedback"
        
        
        return "END"
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(DataAnalysisState)
        workflow.add_node("llm_routing", self._llm_routing_node)
        workflow.add_node("get_data_info", self.get_data_info)
        workflow.add_node("generate_pandas_code", self.generate_pandas_code)
        workflow.add_node("execute_python_code", self.execute_python_code)
        workflow.add_node("error_handling", self.error_handling)
        workflow.add_node("validate_result",self.validate_result)
        workflow.add_node("enhance_query_with_feedback",self.enhance_query_with_feedback)

        workflow.set_entry_point("llm_routing")
        workflow.add_conditional_edges(
            "llm_routing", 
            self._route_condition, {
                "get_data_info": "get_data_info",
                "execute_python_code": "execute_python_code",
                "END": END
            })

        workflow.add_conditional_edges(
            "get_data_info",
            self._route_condition, {
                "END": END,
                "generate_pandas_code": "generate_pandas_code",
                "error_handling": "error_handling"
            })

        workflow.add_conditional_edges(
            "generate_pandas_code",
            self._route_condition, {
                "validate_result": "validate_result",
                "execute_python_code": "execute_python_code",
                "error_handling": "error_handling"
            })

        workflow.add_conditional_edges(
            "execute_python_code",
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

        workflow.add_edge("enhance_query_with_feedback","generate_pandas_code")

        workflow.add_conditional_edges(
            "error_handling",
            self._error_route_condition, {
                "END": END,
                "get_data_info": "get_data_info",
                "generate_pandas_code": "generate_pandas_code", 
            })
        
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    

    def process(self, query: str = None) -> AgentResult:
        if query:
            self.query = query
            
        logger.info(f"Processing data analysis query: {self.query}")

        try:
            config = {"configurable": {"thread_id": "data_analysis_thread"}}
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
                "generated_code": final_state.get("generated_code"),
                "execution_result": final_state.get("execution_result"),
                "data_info": final_state.get("data_info")
            }

            result_df = None
            if self.the_answer:
                result_df = self.the_answer.dataframe
            
            logger.info("Data analysis agent processing completed successfully")
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent": self.agent_name,
                    "query": self.query,
                    "doc_path": self.doc_path,
                    "column_info_path": self.column_info_path,
                    "dataframe": result_df,
                }
            )
            
        except Exception as e:
            error_msg = f"Data analysis agent processing failed: {str(e)}"
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
    """Example usage"""
    import os
    
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            print("GROQ_API_KEY not found")
            return
        
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
        
        df = pd.read_csv("Data/goalscorers.csv")
        
        agent = DataAnalysisAgent(
            llm=llm,
            df=df
        )
        
        test_queries = [
            "get me the top 5 countries which has most goals from penalty"
        ]
        
        for query in test_queries:
            print(f"\nTesting: {query}")
            result = agent.process(query)
            if result.success:
                print("Code:\n", result.data["generated_code"])
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