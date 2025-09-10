import os
from typing import Optional
from langchain_groq import ChatGroq
import pandas as pd
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.tools import Tool
from pydantic import BaseModel, Field
from utils.document import DocumentProcessor
from utils.loggerCenter import LoggerCenter
from utils.base_agent import  BaseSpecializedAgent,  to_dataframe_safe,AgentResult
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import StructuredTool

from utils.vectordeneme import ContextFind

logger = LoggerCenter().get_logger()

class CodeOutput(BaseModel):
    """Pydantic model for structured code output"""
    code: str = Field(description="Python pandas/SQL code that assigns the result to 'result' variable.")
    explanation: Optional[str] = Field(description="Brief explanation of what the code does", default=None)
    data_source: Optional[str] = Field(description="Source of data (csv, postgres, api)", default="csv")

class PythonExecutionResult(BaseModel):
    success: bool
    dataframe: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None
    formatted_output: str = ""

    class Config:
        arbitrary_types_allowed = True

class DataAnalysisAgent(BaseSpecializedAgent):
    """Specialized agent for data analysis tasks"""
    
    def __init__(self, llm: ChatGroq, df: pd.DataFrame,doc_path:str=None,column_info_path:str=None):
        self.df = df
        self.doc_path=doc_path
        self.column_info_path=column_info_path
        self.parser = PydanticOutputParser(pydantic_object=CodeOutput)

        super().__init__("DataAnalysis", llm)

        self.the_answer=None
        self.data_info=None
        self.data_info_loaded = False

        self.context_finder=None
        if self.doc_path is not None:
            try:
                self.context_finder=ContextFind(doc_path)
                logger.info("Context finder has been set")
            except Exception as e:
                logger.error(f"Error while starting context finder: {e}")
            
        self.columnInfo=None
        if self.columnInfo is not None:

            try:
                self.columnInfo=DocumentProcessor().extract_text_from_documents(column_info_path)
                logger.info("Column info uploaded")
            except:
                logger.error(f"Error while starting column info: {e}")

    def _get_system_prompt(self) -> str:
        """System prompt with safe sample data (first 5 rows)"""
        logger.info("Getting system prompt")
        
        data_context = ""
        if self.df is not None:
            
            num_rows = self.df.shape[0]
            num_cols = self.df.shape[1]
            
            sample_df = self.df.head(3)
            sample_str = sample_df.to_string(index=False)
            safe_sample = sample_str.replace('{', '[').replace('}', ']')
            
            data_context = f"""
CURRENT DATA CONTEXT:
- Data available: {num_rows} rows x {num_cols} columns
- Sample data:
{safe_sample}
"""
            
        else:
            data_context = "\nCURRENT DATA CONTEXT:\n- Data is available: NO"
        
        return f"""You are a specialized data analysis expert. Your role is to help analyze data using the available tools.

AVAILABLE TOOLS:
1. data_summary: Get comprehensive data overview (columns, statistics, sample, column infos , context/meaning of the columns)
2. generate_pandas_code: Generate pandas code from natural language query
3. execute_python_code: Execute pandas code safely on the DataFrame

IMPORTANT INSTRUCTIONS and WORKFLOW:
- You have access to a DataFrame with real data (see sample below)
- When user asks about data structure/columns/info/context use data_summary for complete overview
  * Use data summary tool
- When user wants specific analysis/filtering/calculations:
  *You SHOULD start by calling get_data_summary first to load all necessary information
  *Then handle the user request, you can either
  * Use generate_pandas_code to generate the code first, then execute_python_code to run it
  * Or directly use execute_python_code if you know the pandas code
- Always assign results to a variable named 'result' when using execute_python_code
- Be helpful and provide clear explanations of the results


VERY IMPORTANT:
-IF CODE GENERATING TOOL (generate_pandas_code) IS USED RETURN CODE ONLY

{data_context}

Your goal is to help users understand and analyze their data effectively!
"""

    def _generating_pandas_code(self) -> str:
        """Generate pandas code for the given query"""
        template = """
You are an expert data analyst. You analyze the question STEP BY STEP and generate a Python pandas code that answers the question using the provided DataFrame.


YOUR DATA CONTEXT:
{data_info}


IMPORTANT:
1. Do NOT include anything outside of CODE and an Explanation.
2. Explanation MUST be about what is understood from the QUESTION.
Your response must be exactly this format:
{{"code": "your_pandas_code_here", "explanation": "brief explanation"}}

{format_instructions}

CRITICAL REQUIREMENTS:
1. Use ONLY pandas operations - pandas and numpy are already imported as 'pd' and 'np'
2. UNDERSTAND what input wants exactly
3. The DataFrame is already loaded as 'df'
4. Do NOT include any import statements in your code

Question: {query}
"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["query", "data_info"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        chain = prompt | self.llm | self.parser
        
        logger.info(f"Generating code for query: {self.query}")
                        
        try:
            response = chain.invoke({ "query": self.query, "data_info":self.data_info })
            
            if isinstance(response, dict):
                if "text" in response:
                    code = response["text"].code if hasattr(response["text"], 'code') else str(response["text"])
                else:
                    code = response.get("code", str(response))
            else:
                code = response.code if hasattr(response, 'code') else str(response)
            
            logger.info(f"Code generated: {code} \n\n Response:{response}")
            return code
            
        except Exception as e:
            logger.error(f"Error in code generation: {e}")
            return f"Error generating code: {str(e)}"
    
    def _is_safe_code(self, code: str) -> bool:
        """Check if code is safe to execute"""
        dangerous_keywords = [
            'import os', 'import sys', 'import subprocess', 'import shutil',
            'exec(', 'eval(', 'open(', 'file(', 'input(', 'raw_input(',
            '__import__', 'globals()', 'locals()', 'dir()', 'delattr',
            'setattr', 'getattr', 'hasattr', 'exit(', 'quit()'
        ]
        
        code_lower = code.lower()
        for keyword in dangerous_keywords:
            if keyword in code_lower:
                logger.warning(f"Potentially dangerous code detected: {keyword}")
                return False
        return True

    def _setup_tools(self):
        
        logger.info("Setting up tools")
        class DataInfoInput(BaseModel):
            query: Optional[str] = Field(default="", description="Optional user query to get relevant context and data info")
        
        class PythonGenerationInput(BaseModel):
            query: str = Field(description="User request for python code generation")
        
        class PythonExecutionInput(BaseModel):
            pandas_code: str = Field(description="Python code to execute")
        
        def get_data_summary(query: str = "") -> str:
            """Get comprehensive data summary"""
            logger.info(f"Getting comprehensive data summary for query: {query}")

            if self.df is None:
                return "No data loaded"
            
            context_info="No data available"
            columnInfo_info="No data available"
            doc_link = "No source available"
            columnInfo_link = "No source available"
            
            try:
                shape_info = f"{self.df.shape[0]} rows x {self.df.shape[1]} columns"
                columns_list = [str(col).replace('{', '[').replace('}', ']') for col in self.df.columns]
                columns = ", ".join(columns_list)
                d_types=dict(self.df.dtypes)

                if self.context_finder and query:
                    try:
                        context_info = self.context_finder.return_context(query, top_k=3)
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

                self.data_info= f"""
Data Summary:
- Shape: {shape_info}
- Columns: {columns}
- Data Types: {d_types}


Relevant Context:
{context_info}

Column info:
{columnInfo_info}

Source of context={doc_link}
Source of cocolumn info text={columnInfo_link}

"""
                self.database_info_loaded=True
                return self.data_info
            
            except Exception as e:
                logger.error(f"Error getting data summary: {e}")
            

        def generate_pandas_code(query: str = "") -> str: 
            """Generate pandas code here"""
            logger.info(f"Generate pandas code for query: {query}")

            if not self.data_info_loaded or not self.data_info:
                    logger.info("Database info not loaded, loading automatically...")
                    get_data_summary(query)

            try:
                if query:
                    self.query = query
                return self._generating_pandas_code()
            except Exception as e:
                self.log_error(f"Code generation failed: {e}")
                return f"Code generation failed: {str(e)}"

        def execute_python_code(pandas_code: str) -> str:
            """Execute pandas code"""
            logger.info(f"Executing pandas code: {pandas_code}")
            
            if self.df is None:
                return "No data available for code execution"
            
            try:
                if not self._is_safe_code(pandas_code):
                    return "Code is dangerous"
                
                logger.info("Executing code in safe environment")
                
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
                        'print': print
                    }
                }
                
                local_vars = {}
                exec(pandas_code, safe_globals, local_vars)
                
                result = local_vars.get("result", None)
                
                if result is not None:
                    logger.info("Code executed successfully")

                    if not isinstance(result, pd.DataFrame):
                        result=to_dataframe_safe(result)
                    
                    if isinstance(result, pd.DataFrame):
                        if result.empty:
                            formatted_output= "Code executed successfully. Result: Empty DataFrame"
                            
                            self.the_answer= PythonExecutionResult(
                                success=True,
                                dataframe=result,
                                formatted_output= formatted_output
                            )
                        
                            return formatted_output
                    

                        if len(result) > 20:
                            display_result = result.head(20)


                            formatted_output= f"Code executed successfully. Result (showing first 20 rows of {len(result)}):\n{display_result.to_string()}"
                            
                            self.the_answer= PythonExecutionResult(
                                success=True,
                                dataframe=result,
                                formatted_output= formatted_output
                            )
                        
                            return formatted_output

                        else:
                            formatted_output= f"Code executed successfully. Result:\n{result.to_string()}"
                            
                            self.the_answer= PythonExecutionResult(
                                success=True,
                                dataframe=result,
                                formatted_output= formatted_output
                            )
                        
                            return formatted_output

                    else:
                        return f"Code executed successfully. Result: {result}"
                else:
                    return "Code executed successfully but no 'result' variable was assigned"
                
            except Exception as e:
                import traceback
                error_msg = f"Code execution failed: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                logger.error(error_msg)
                return error_msg
                
        self.tools = [
            StructuredTool.from_function(
                name="data_summary", 
                description="Get data summary and statistics including shape, columns, data types, column info and context", 
                func=get_data_summary,
                args_schema=DataInfoInput
            ),
            StructuredTool.from_function(
                name="generate_pandas_code", 
                description="Generate pandas code from natural language query to analyze the DataFrame", 
                func=generate_pandas_code,
                args_schema=PythonGenerationInput
            ),
            StructuredTool.from_function(
                name="execute_python_code",
                description="Execute pandas code safely on the DataFrame. The code should assign results to a variable named 'result'",
                func=execute_python_code,
                args_schema=PythonExecutionInput
            )
        ]

    def process(self, query: str) -> AgentResult:
        """Process query through this specialized agent"""
        logger.info(f"Processing query: {query}")
        
        try:
            self.query = query

            response = self.agent_executor.invoke({"input": query})
            
            return AgentResult(
                success=True,
                data=response,
                metadata={"agent": self.agent_name, "query": query, "dataframe":self.the_answer.dataframe}
            )
        
        except Exception as e:
            error_msg = f"[{self.agent_name}] Error processing query '{query}': {str(e)}"
            logger.error(error_msg)
            return AgentResult(success=False, error=error_msg)

    
def main():
    query = "string"
    while query != "-1":
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not found")
            
            llm = ChatGroq(
                model_name="llama-3.1-8b-instant",  
                api_key=groq_api_key,
                temperature=0.1
            )
        
            df = pd.read_csv("Data/goalscorers.csv")
            
            query = "Get me a list of 10 players who scored most goals"
            if query == "-1":
                break
            
            system = DataAnalysisAgent(llm=llm, df=df)
            
            result = system.process(query)
            result_info=result.data["output"]

            if result.metadata.get("dataframe") is not None:
                dataframe = result.metadata["dataframe"]     

            
            if result.success:
                print(f"Result: {result_info}\n\n")
                print(f"*"*50)
                print(f"{dataframe}")
            else:
                print(f"Error: {result.error}")
            
        except Exception as e:
            logger.error(f"Application error: {e}")
            print(f"Application error: {e}")

if __name__ == "__main__":
    main()