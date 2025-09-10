import os
from typing import Any, Dict, Literal, Optional, TypedDict
from langchain_groq import ChatGroq
from utils.loggerCenter import LoggerCenter
from utils.base_agent import BaseSpecializedAgent, AgentResult, clean_imports_from_code, is_safe_code
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import pandas as pd
import numpy as np
from utils.vectordeneme import ContextFind  
from utils.document import DocumentProcessor
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from enum import Enum
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import glob

logger = LoggerCenter().get_logger()

#############################################################################
class VisualizationErrorCorrection(BaseModel):
    next_node: str = Field(description="Next node to execute: 'get_data_info', 'generate_plot_code', or 'END'")
    corrected_user_query: Optional[str] = Field(description="Enhanced user query that includes error context and fix instructions. Always provided for plot generation errors.")
    reasoning: str = Field(description="Explanation of error analysis and query enhancement strategy")
    confidence: float = Field(description="Confidence level between 0.0 and 1.0")
#############################################################################

#################################################################################
class VisualizationValidationResult(BaseModel):
    is_correct: bool = Field(description="Whether the visualization correctly answers the user query")
    accuracy_score: float = Field(ge=0.0, le=1.0, description="Accuracy score of the visualization (0.0 to 1.0)")
    feedback: str = Field(description="Detailed feedback about what's wrong or missing in the visualization")
    improvement_suggestions: str = Field(description="Specific suggestions to improve the query/plot code")
    should_retry: bool = Field(description="Whether the query should be retried with improvements")

class VisualizationQueryEnhancement(BaseModel):
    enhanced_query: str = Field(description="Enhanced query with improvement notes and specific visualization guidance")
    enhancement_reasoning: str = Field(description="Explanation of what was enhanced and why")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the enhancement")
#############################################################################

class PlotExecutionResult(BaseModel):
    success: bool
    code: str
    plot_path: Optional[str] = None
    error_message: Optional[str] = None
    formatted_output: str = ""
    plot_type: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

class VisualizationWorkflowType(str, Enum):
    DATA_INFO_ONLY = "data_info_only"
    GENERATE_PLOT_CODE_ONLY = "generate_plot_code_only"
    CREATE_VISUALIZATION = "create_visualization"
    DIRECT_PLOT = "direct_plot"
    INVALID = "invalid"

class VisualizationRoutingDecision(BaseModel):
    workflow_type: VisualizationWorkflowType = Field(description="The type of visualization workflow to execute")
    reasoning: str = Field(description="Explanation of the routing decision")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in routing decision")

class PlotOutput(BaseModel):
    code: str = Field(description="Matplotlib/seaborn plotting code")
    explanation: str = Field(description="Explanation of the visualization")
    plot_type: str = Field(description="Type of plot being created")

class VisualizationState(TypedDict):
    query: str
    route_decision: Optional[VisualizationRoutingDecision]
    routing_reasoning: Optional[str]
    confidence_score: Optional[float]
    result: Optional[str]
    error: Optional[str]
    metadata: Optional[Dict[str, Any]]
    data_info_loaded: Optional[bool]
    data_info: Optional[str]
    generated_plot_code: Optional[str]
    plot_execution_result: Optional[str]
    plot_path: Optional[str]
    next_error_node: Optional[str]
    retry_count: Optional[int]
    validation_attempts: Optional[int]  # Track validation attempts
    validation_result: Optional[VisualizationValidationResult]
    is_validated: Optional[bool]
    final_result: Optional[str]

class VisualizationAgent(BaseSpecializedAgent):

    def __init__(self, llm: ChatGroq, df: pd.DataFrame, doc_path: str = None, column_info_path: str = None, plots_dir: str = "plots"):
        super().__init__("VisualizationAgent", llm)

        self.df = df
        self.doc_path = doc_path
        self.column_info_path = column_info_path
        self.plots_dir = plots_dir
        self.max_retries = 5
        self.max_validation_attempts = 4
        
        os.makedirs(self.plots_dir, exist_ok=True)
        
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
        self.plot_result = None
        self.data_info = None
        
        plt.style.use('default')
        sns.set_palette("husl")
        warnings.filterwarnings('ignore')
        
        self.app = self._build_graph()

    def _llm_routing_node(self, state: VisualizationState) -> VisualizationState:
        query = state["query"]
        logger.info(f"Making intelligent LLM-based routing decision for visualization: {query}")
        
        parser = PydanticOutputParser(pydantic_object=VisualizationRoutingDecision)
        
        routing_prompt = f"""
You are an intelligent routing system for data visualization tasks.
Your job is to analyze the USER QUERY and decide which visualization operations should be executed.

USER QUERY:
"{query}"

AVAILABLE TOOLS/NODES:
1. get_data_info        → Retrieves DataFrame schema, statistics, context, and column information for visualization planning
2. generate_plot_code   → Generates matplotlib/seaborn plotting code from user requirements and data context
3. create_visualization → Executes plotting code and creates/saves visualizations

ROUTING RULES:
1. If the user asks for data details/summary/info for visualization → classify as "data_info_only"
2. If the user only wants plotting code → classify as "generate_plot_code_only"  
3. If the user wants both code generation and visualization creation → classify as "create_visualization"
4. If the user provides plotting code and asks to create/execute it → classify as "direct_plot"
5. If the user query is not related to data visualization → classify as "invalid"

WORKFLOW PATTERNS:
- "data_info_only": get_data_info → END
- "generate_plot_code_only": get_data_info → generate_plot_code → END
- "create_visualization": get_data_info → generate_plot_code → create_visualization → END
- "direct_plot": create_visualization → END
- "invalid": → END

OUTPUT FORMAT:
You must output only valid JSON matching the schema below.
Do not include explanations, markdown, or code blocks.
Output JSON only.

EXAMPLE (user asks for scatter plot of goals vs assists):
{{
    "workflow_type": "create_visualization",
    "reasoning": "User wants a complete visualization created, need to load data info, generate plotting code, and create the plot",
    "confidence": 0.9
}}

{parser.get_format_instructions()}
"""

        try:
            response = self.llm.invoke(routing_prompt)
            routing_decision = parser.parse(response.content)

            if routing_decision.workflow_type == VisualizationWorkflowType.INVALID:
                return {
                    **state,
                    "route_decision": routing_decision,
                    "error": "Routing classification failed: Query is not suitable for data visualization operations."
                }
            
            return {
                **state,
                "route_decision": routing_decision,
                "routing_reasoning": routing_decision.reasoning,
                "confidence_score": routing_decision.confidence
            }
            
        except Exception as e:
            logger.error(f"LLM routing error: {e}")
            fallback_decision = VisualizationRoutingDecision(
                workflow_type=VisualizationWorkflowType.INVALID,
                reasoning=f"Routing classification failed: Error: {e}",
                confidence=0.8
            )
            
            return {
                **state,
                "route_decision": fallback_decision,
                "error": f"Routing classification failed: Error: {e}"
            }

    def get_data_info(self, state: VisualizationState) -> VisualizationState:
        try:
            logger.info(f"Getting the data info for visualization.")
            if self.df is None:
                state["error"] = "No DataFrame available for visualization"
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

            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

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
VISUALIZATION DATA CONTEXT:
Shape: {num_rows} rows x {num_cols} columns
Columns: {columns}
Data Types: {d_types}
Numeric Columns: {numeric_cols}
Categorical Columns: {categorical_cols}

Sample Data (first 3 rows):
{safe_sample}

RELEVANT CONTEXT:
{context_info}

COLUMN INFORMATION:
{column_info_text}

Context retrieved from: {doc_link}
Column Info retrieved from: {column_info_link}

VISUALIZATION RECOMMENDATIONS:
- Use numeric columns for: histograms, box plots, scatter plots, line plots
- Use categorical columns for: bar charts, count plots, pie charts
- For relationships: scatter plots between numeric columns, grouped bar charts
- For distributions: histograms, density plots, violin plots
- For time series: line plots if datetime columns exist
"""
            logger.info(f"Visualization Data Info uploaded: {self.data_info}")

            state["data_info_loaded"] = True
            state["data_info"] = self.data_info
            return state

        except Exception as e:
            state["error"] = f"Data info retrieval failed: {e}"
            logger.error(state["error"])
            return state

    def generate_plot_code(self, state: VisualizationState) -> VisualizationState:
        try:
            query = state["query"]
            logger.info(f"Plot code is being generated. User query -> {query}")

            if not state["data_info_loaded"]:
                logger.info("Data info not loaded, loading automatically...")
                self.get_data_info(state)
            
            parser = PydanticOutputParser(pydantic_object=PlotOutput)
            
            words = query.strip().split()
            first_three = words[:3]
            short_query = "".join(first_three)
            
            template = """
You are an expert data visualization specialist. Generate matplotlib/seaborn plotting code for the given request using the loaded data information.

=== LOADED DATA INFORMATION ===
{data_info}

=== VISUALIZATION CODE INSTRUCTIONS ===
1. Use the DataFrame schema and context information provided above
2. Choose appropriate plot types based on data types and user request
3. Do NOT include anything outside of CODE, EXPLANATION, and PLOT_TYPE

OUTPUT FORMAT:
You must output only valid JSON matching the schema below.
Do not include explanations, markdown, or code blocks.
Output JSON only.
{{
  "code": "your_plotting_code_here",
  "explanation": "brief explanation of the visualization",
  "plot_type": "type_of_plot_created"
}}

{format_instructions}

CRITICAL REQUIREMENTS:
1. Generate ONLY matplotlib/seaborn plotting code without markdown formatting
2. Use proper column names from the loaded schema above
3. The DataFrame is already available as 'df'
4. matplotlib.pyplot is available as 'plt', seaborn is available as 'sns'
5. pandas is available as 'pd', numpy is available as 'np'
6. Do NOT include any import statements
7. Always use figure size: plt.figure(figsize=(12, 8)) or similar
8. Always end with: plt.savefig('{plots_dir}/plot_{short_query}.png', dpi=300, bbox_inches='tight')
9. Include plt.tight_layout() for better spacing
10. Add proper titles, labels, and styling
11. Handle missing values appropriately
12. Ensure the code is safe (no file operations except plt.savefig, no system calls, etc.)

AVAILABLE PLOT TYPES:
- Bar plots: For categorical data comparison
- Line plots: For time series or trends
- Histograms: For numerical distributions  
- Scatter plots: For relationships between variables
- Box plots: For distribution comparisons
- Heatmaps: For correlation matrices
- Violin plots: For detailed distributions
- Count plots: For categorical frequencies
- Subplots: For multiple related visualizations

User Visualization Request: {user_request}
"""
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["user_request", "data_info", "plots_dir", "short_query"],
                partial_variables={"format_instructions": parser.get_format_instructions}
            )
            
            chain = prompt | self.llm | parser
            
            response = chain.invoke({
                "user_request": query,
                "data_info": self.data_info,
                "plots_dir": self.plots_dir,
                "short_query": short_query
            })
            
            try:
                if isinstance(response, dict):
                    if "text" in response:
                        plot_code = response["text"].code if hasattr(response["text"], 'code') else str(response["text"])
                    else:
                        plot_code = response.get("code", str(response))
                else:
                    plot_code = response.code if hasattr(response, 'code') else str(response)
                
                logger.info(f"Plot code generated: {plot_code}")
                state["generated_plot_code"] = plot_code
                return state
                
            except Exception as e:
                state["error"] = f"Plot code generation failed: Error while cleaning response {e}"
                logger.error(state["error"])
                return state
                                
        except Exception as e:
            state["error"] = f"Plot code generation failed: Error while producing response {e}"
            logger.error(state["error"])
            return state

    def create_visualization(self, state: VisualizationState) -> VisualizationState:
        
        if not state["generated_plot_code"] and state["route_decision"].workflow_type != VisualizationWorkflowType.DIRECT_PLOT:
            logger.info("Plot code not generated, generating automatically...")
            self.generate_plot_code(state)
        
        if state["route_decision"].workflow_type == VisualizationWorkflowType.DIRECT_PLOT:
            state["generated_plot_code"] = state["query"]
        
        try:
            plot_code = state["generated_plot_code"]
            logger.info(f"Creating visualization with code")

            if self.df is None:
                state["error"] = "No DataFrame available for visualization creation"
                return state
            
            if not is_safe_code(plot_code):
                state["error"] = "Plot code contains potentially dangerous operations and cannot be executed"
                logger.error(state["error"])
                return state

            plt.clf()
            plt.close('all')

            safe_globals = {
                'pd': pd,
                'np': np,
                'df': self.df.copy(),
                'plt': plt,
                'sns': sns,
                'os': os,
                'warnings': warnings,
                '__builtins__': {
                    'len': len, 'str': str, 'int': int, 'float': float,
                    'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                    'range': range, 'enumerate': enumerate, 'zip': zip,
                    'sorted': sorted, 'reversed': reversed, 'sum': sum,
                    'min': min, 'max': max, 'abs': abs, 'round': round,
                    'print': print, 'isinstance': isinstance, 'hasattr': hasattr,'__import__': __import__ 
                }
            }
            
            local_vars = {}
            exec(plot_code, safe_globals, local_vars)
            
            plot_files = (glob.glob("*.png") + glob.glob("*.jpg") + 
                         glob.glob("*.jpeg") + glob.glob(f"{self.plots_dir}/*.png") +
                         glob.glob(f"{self.plots_dir}/*.jpg") + glob.glob(f"{self.plots_dir}/*.jpeg"))
            
            if plot_files:
                plot_path = max(plot_files, key=os.path.getctime)
                
                result_text = f"""Visualization Created Successfully!

Code: {plot_code}

Plot saved to: {plot_path}
"""
                
                state["plot_execution_result"] = f"Visualization created successfully: {plot_path}"
                state["plot_path"] = plot_path
                
                self.plot_result = PlotExecutionResult(
                    success=True,
                    code=plot_code,
                    plot_path=plot_path,
                    formatted_output=result_text
                )
                
                logger.info(f"Visualization created successfully: {plot_path}")
                plt.close('all')
                return state
            else:
                state["error"] = "Visualization code executed but no plot file was created. Make sure to use plt.savefig() in your code."
                logger.error(state["error"])
                plt.close('all')
                return state

        except Exception as e:
            import traceback
            error_msg = f"Visualization creation failed: \nCode: {plot_code}\n\nTraceback:\n{traceback.format_exc()}"
            state["error"] = error_msg
            logger.error(state["error"])
            plt.close('all')
            return state

#######################################################################################

    def validate_result(self, state: VisualizationState) -> VisualizationState:
        
        try:
            original_query = state.get("query", "")
            generated_code = state.get("generated_plot_code", "")
            validation_attempts = state.get("validation_attempts", 0)
            
            logger.info(f"Validating plot code (attempt {validation_attempts + 1}/{self.max_validation_attempts})")

            if validation_attempts >= self.max_validation_attempts:
                return state
            
            parser = PydanticOutputParser(pydantic_object=VisualizationValidationResult)
            
            validation_prompt = f"""
You are an expert data visualization code validator. Your task is to evaluate whether the generated plot CODE correctly addresses the user's original query by analyzing the code logic and approach.

IMPORTANT: You cannot see the actual plot image. Evaluate ONLY based on the CODE and whether it logically should produce the requested visualization.

=== ORIGINAL USER QUERY ===
"{original_query}"

=== GENERATED PLOT CODE ===
{generated_code}

=== DATA CONTEXT ===
{self.data_info if self.data_info else "No data info available"}

=== CODE VALIDATION CRITERIA ===
1. Does the code logically address what the user asked for?
2. Is the plot type (bar, scatter, histogram, etc.) appropriate for the query?
3. Are the correct column names used based on available data?
4. Is the data processing logic correct (groupby, filtering, aggregation)?
5. Are proper labels, titles, and formatting included in the code?
6. Does the code handle potential data issues (missing values, data types)?
7. Is the matplotlib/seaborn syntax correct and complete?
8. Does the code save the plot properly with plt.savefig()?

=== ACCURACY SCORING GUIDE FOR CODE ===
- 1.0: Perfect code
- 0.8-0.9: Good with minor issues  
- 0.6-0.7: Partial with gaps
- 0.4-0.5: Poor with major issues
- 0.0-0.3: Wrong approach

=== CRITICAL RULES ===
- If accuracy_score >= 0.8, set is_correct = true and should_retry = false
- If accuracy_score < 0.8, set is_correct = false and should_retry = true
- Focus on CODE LOGIC and DATA HANDLING, not visual aesthetics you cannot see
- Provide specific feedback about code improvements (column names, plot types, data processing)
- Check if the code would logically produce what the user requested

REQUIRED JSON OUTPUT (no additional text):
{{
    "is_correct": boolean,
    "accuracy_score": float_between_0_and_1,
    "feedback": "detailed_feedback_about_code_logic_and_data_handling_issues",
    "improvement_suggestions": "specific_suggestions_to_fix_code_logic_and_approach",
    "should_retry": boolean
}}

{parser.get_format_instructions()}
"""
            
            response = self.llm.invoke(validation_prompt)
            validation_result = parser.parse(response.content)
            
            state["validation_result"] = validation_result
            state["validation_attempts"] = validation_attempts + 1
            
            logger.info(f"Plot code validation result: accuracy={validation_result.accuracy_score:.2f}, correct={validation_result.is_correct}")
            
            return state
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            fallback_validation = VisualizationValidationResult(
                is_correct=True,
                accuracy_score=0.8,
                feedback=f"Validation process failed: {e}",
                improvement_suggestions="Manual review recommended",
                should_retry=False
            )
            state["validation_result"] = fallback_validation
            state["validation_attempts"] = state.get("validation_attempts", 0) + 1
            return state

    def enhance_query_with_feedback(self, state: VisualizationState) -> VisualizationState:

        try:
            original_query = state.get("query", "")
            validation_result = state.get("validation_result")
            current_code = state.get("generated_plot_code", "")
            
            parser = PydanticOutputParser(pydantic_object=VisualizationQueryEnhancement)
            
            enhancement_prompt = f"""
You are a query enhancement expert for data visualization.

ORIGINAL QUERY: "{original_query}"
Generated Code: {current_code}
VALIDATION FEEDBACK:
- Score: {validation_result.accuracy_score}
- Issues: {validation_result.feedback}
- Suggestions: {validation_result.improvement_suggestions}

DATA: {self.data_info if self.data_info else "Not available"}

TASK: Add System Validation Note to original query with specific fixes.

Example:
Original: "Show goals by team"
Issues: "Missing 'goals' column, use 'total_goals'"
Output: "Show goals by team.

System Validation Note: Use 'total_goals' column, group by 'team', add proper labels and sort descending."

REQUIRED JSON (no extra text):
{{
    "enhanced_query": "original_query + system_validation_note",
    "enhancement_reasoning": "brief_changes_explanation", 
    "confidence": 0.0-1.0
}}
"""
            
            response = self.llm.invoke(enhancement_prompt)
            enhancement = parser.parse(response.content)
            
            state["query"] = enhancement.enhanced_query
            logger.info(f"Query enhanced: {enhancement.enhanced_query}")

            state["validation_result"] = None
            state["generated_plot_code"] = None
            state["plot_execution_result"] = None
            state["plot_path"] = None
            state["result"] = None
            
            return state
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            fallback_query = state["query"] + "\nNotes: " + state["validation_result"].feedback
            state["query"] = fallback_query
            logger.info(f"Query enhanced(fallback): {fallback_query}")
            state["validation_result"] = None
            state["generated_plot_code"] = None
            state["plot_execution_result"] = None
            state["plot_path"] = None
            state["result"] = None

            return state
    
    #######################################################################
    
    def error_handling(self, state: VisualizationState) -> VisualizationState:
        error = state.get("error", "Unknown error occurred")
        retry_count = state.get("retry_count", 0)
        
        logger.error(f"Error in visualization workflow (retry {retry_count}/{self.max_retries})")
        
        if retry_count >= self.max_retries:
            logger.error(f"Maximum retry limit ({self.max_retries}) exceeded")
            return self._simple_error_handling(state)
        
        try:
            return self._llm_error_correction(state)
        except Exception as e:
            logger.error(f"LLM error correction failed: {e}")
            return self._simple_error_handling(state)

    def _llm_error_correction(self, state: VisualizationState) -> VisualizationState:
        error = state.get("error", "Unknown error occurred")
        query = state.get("query", "")
        plot_code = state.get("generated_plot_code", "")
        current_node = self._get_current_node_context(state)
        
        parser = PydanticOutputParser(pydantic_object=VisualizationErrorCorrection)
        
        error_handling_prompt = f"""
You are an error handler for data visualization workflows.

CURRENT ERROR:
- Failed Node: {current_node}
- User Query: "{query}"
- Error: "{error}"
- Retry: {state.get("retry_count", 0)}/{self.max_retries}

DATA INFO: {self.data_info if self.data_info else "Not loaded"}

RULES:
1. Data/schema errors → next_node: "get_data_info"
2. Code syntax/logic errors → Fix and next_node: "generate_plot_code"  
3. Too many retries → next_node: "END"

OUTPUT FORMAT:
Original query + System Error Note with fix instructions.

Example:
Original: "Create bar chart of sales"
Error: KeyError 'sales_amount'
Output: "Create bar chart of sales.

System Error Note: Use 'total_sales' column instead of 'sales_amount'. Apply proper matplotlib syntax."

REQUIRED JSON (no extra text):
{{
    "next_node": "get_data_info|generate_plot_code|END",
    "corrected_user_query": "original_query + system_error_note",
    "reasoning": "brief_fix_explanation",
    "confidence": 0.0-1.0
}}
"""

        response = self.llm.invoke(error_handling_prompt)
        correction = parser.parse(response.content)
        
        valid_nodes = ["get_data_info", "generate_plot_code", "END"]
        if correction.next_node not in valid_nodes:
            logger.warning(f"Invalid next_node '{correction.next_node}', defaulting to END")
            correction.next_node = "END"
        
        state["retry_count"] = state.get("retry_count", 0) + 1
        
        if correction.corrected_user_query and correction.corrected_user_query != query:
            state["query"] = correction.corrected_user_query
            logger.info(f"Query enhanced with error context: {correction.corrected_user_query}")
        
        if correction.next_node == "generate_plot_code":
            state["generated_plot_code"] = None
            logger.info("Cleared generated_plot_code to force regeneration with enhanced query")
        
        state["error"] = None
        state["next_error_node"] = correction.next_node
        
        logger.info(f"Error correction applied.")
        logger.info(f"Next node to execute: {correction.next_node}")
        return state

    def _simple_error_handling(self, state: VisualizationState) -> VisualizationState:
        """Fallback simple error handling"""
        error = state.get("error", "Unknown error occurred")
        
        if "Visualization creation failed" in error:
            state["result"] = f"Plot creation failed.\n\nDetails: {error}"
        elif "generation failed" in error:
            state["result"] = f"Could not generate plotting code.\n\nDetails: {error}"
        elif "Data info retrieval failed" in error:
            state["result"] = f"Could not load data information for visualization.\n\nDetails: {error}"
        elif "not suitable for data visualization" in error:
            state["result"] = f"This query is not related to data visualization operations.\n\nPlease ask visualization questions."
        else:
            state["result"] = f"A visualization error occurred: {error}"
        
        state["next_error_node"] = "END"
        return state

    def _error_route_condition(self, state: VisualizationState) -> Literal["get_data_info", "generate_plot_code", "create_visualization", "END"]:
        next_node = state.get("next_error_node", "END")
        logger.info(f"Error handling routing to: {next_node}")
        return next_node
    
    #######################################################################
    def _get_current_node_context(self, state: VisualizationState) -> str:
        if not state.get("data_info_loaded"):
            return "get_data_info"
        elif not state.get("generated_plot_code"):
            return "generate_plot_code" 
        elif not state.get("plot_execution_result"):
            return "create_visualization"
        else:
            return "unknown"
    
    def _route_condition(self, state: VisualizationState) -> Literal["enhance_query_with_feedback","validate_result","get_data_info", "generate_plot_code", "create_visualization", "error_handling", "END"]:
        route = state.get("route_decision")

        if state.get("error"):
            logger.info(f"-->error_handling")            
            return "error_handling"
        
        if not route:
            logger.info(f"-->error_handling")
            return "error_handling"
        
        if route.workflow_type == VisualizationWorkflowType.INVALID:
            logger.info(f"INVALID -->END")
            return "END"
        
        if not state.get("data_info_loaded") and not state.get("generated_plot_code") and not state.get("plot_execution_result"):
            if route.workflow_type == VisualizationWorkflowType.DIRECT_PLOT:
                logger.info(f"START-->create_visualization")
                return "create_visualization"
            else:
                logger.info(f"START-->get_data_info")
                return "get_data_info"
        
        if state.get("data_info_loaded") and not state.get("generated_plot_code") and not state.get("plot_execution_result"):
            if route.workflow_type == VisualizationWorkflowType.DATA_INFO_ONLY:
                logger.info(f"get_data_info-->END")
                state["result"] = state["data_info"]
                return "END"
            else:
                logger.info(f"get_data_info-->generate_plot_code")
                return "generate_plot_code"
        
        if state.get("generated_plot_code") and not state.get("plot_execution_result"):
            if route.workflow_type == VisualizationWorkflowType.GENERATE_PLOT_CODE_ONLY:
                logger.info(f"generate_plot_code-->validate_result")
                state["result"] = state["generated_plot_code"]
                return "validate_result"
            else:
                logger.info(f"generate_plot_code-->create_visualization")
                return "create_visualization"
            
        if state.get("plot_execution_result") and not state.get("validation_result"):
            state["result"] = state["plot_execution_result"]
            return "validate_result"
        
        if state.get("validation_attempts") >= self.max_validation_attempts:
            return "END"
        
        if state.get("validation_result"):
            if state["validation_result"].accuracy_score >= 0.8:
                return "END"
            else:
                return "enhance_query_with_feedback"
        
        return "END"

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(VisualizationState)
        workflow.add_node("llm_routing", self._llm_routing_node)
        workflow.add_node("get_data_info", self.get_data_info)
        workflow.add_node("generate_plot_code", self.generate_plot_code)
        workflow.add_node("create_visualization", self.create_visualization)
        workflow.add_node("error_handling", self.error_handling)
        workflow.add_node("validate_result", self.validate_result)
        workflow.add_node("enhance_query_with_feedback", self.enhance_query_with_feedback)

        workflow.set_entry_point("llm_routing")
        workflow.add_conditional_edges(
            "llm_routing", 
            self._route_condition, {
                "get_data_info": "get_data_info",
                "create_visualization": "create_visualization",
                "END": END
            })

        workflow.add_conditional_edges(
            "get_data_info",
            self._route_condition, {
                "END": END,
                "generate_plot_code": "generate_plot_code",
                "error_handling": "error_handling"
            })

        workflow.add_conditional_edges(
            "generate_plot_code",
            self._route_condition, {
                "validate_result": "validate_result",
                "create_visualization": "create_visualization",
                "error_handling": "error_handling"
            })

        workflow.add_conditional_edges(
            "create_visualization",
            self._route_condition, {
                "validate_result": "validate_result",
                "error_handling": "error_handling"
            })

        workflow.add_conditional_edges(
            "validate_result",
            self._route_condition,{
                "END": END,
                "enhance_query_with_feedback": "enhance_query_with_feedback"
            }
        )

        workflow.add_edge("enhance_query_with_feedback", "generate_plot_code")

        workflow.add_conditional_edges(
            "error_handling",
            self._error_route_condition, {
                "END": END,
                "get_data_info": "get_data_info",
                "generate_plot_code": "generate_plot_code", 
                "create_visualization": "create_visualization"
            })
        
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def process(self, query: str = None) -> AgentResult:
        if query:
            self.query = query
            
        logger.info(f"Processing visualization query: {self.query}")

        try:
            config = {"configurable": {"thread_id": "visualization_thread"}}
            initial_state = {"query": self.query}
            final_state = self.app.invoke(initial_state, config)
            
            if final_state.get("error") and not final_state.get("result"):
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
                "generated_plot_code": final_state.get("generated_plot_code"),
                "plot_execution_result": final_state.get("plot_execution_result"),
                "plot_path": final_state.get("plot_path"),
                "data_info": final_state.get("data_info")
            }

            plot_path = None
            if self.plot_result:
                plot_path = self.plot_result.plot_path
            elif final_state.get("plot_path"):
                plot_path = final_state["plot_path"]
            
            logger.info("Visualization agent processing completed successfully")
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent": self.agent_name,
                    "query": self.query,
                    "doc_path": self.doc_path,
                    "column_info_path": self.column_info_path,
                    "plot_path": plot_path,
                    "plot_success": self.plot_result.success if self.plot_result else True,
                    "plots_dir": self.plots_dir
                }
            )
            
        except Exception as e:
            error_msg = f"Visualization agent processing failed: {str(e)}"
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
        
        llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)
        
        df = pd.read_csv("Data/goalscorers.csv")
        
        agent = VisualizationAgent(
            llm=llm,
            df=df,
            plots_dir="plots"
        )
        
        test_queries = [
            "bring me a plot that shows the top 10 nations who had most own goals on their side"
        ]
        
        for query in test_queries:
            print(f"\nTesting: {query}")
            result = agent.process(query)
            if result.success:
                print("Success!")
                if result.data["generated_plot_code"]:
                    print("Generated Code:\n", result.data["generated_plot_code"])
                if result.metadata.get("plot_path"):
                    print("Plot saved to:", result.metadata["plot_path"])
                print("Result:\n", result.data["output"])
            else:
                print("Error:", result.error)
            print("-" * 50)
            
    except Exception as e:
        print(f"Example failed: {e}")


if __name__ == "__main__":
    example_usage()