import os
from langchain_groq import ChatGroq
import pandas as pd
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.tools import Tool
from loggerCenter import LoggerCenter
from utils import AgentResult, BaseSpecializedAgent, CodeOutput
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

logger = LoggerCenter().get_logger()

class DataAnalysisAgent(BaseSpecializedAgent):
    """Specialized agent for data analysis tasks"""
    
    def __init__(self, llm: ChatGroq, df: pd.DataFrame, query):
        self.df = df
        self.query = query
        self.parser = PydanticOutputParser(pydantic_object=CodeOutput)
        super().__init__("DataAnalysis", llm)

    def _get_system_prompt(self) -> str:
        """System prompt with safe sample data (first 5 rows)"""
        logger.info("Getting system prompt")
        
        data_context = ""
        if self.df is not None:
            try:
                num_rows = self.df.shape[0]
                num_cols = self.df.shape[1]
                
                sample_df = self.df.head(3)
                sample_str = sample_df.to_string(index=False)
                # Curly braces'leri güvenli karakterlerle değiştir
                safe_sample = sample_str.replace('{', '[').replace('}', ']')
                
                data_context = f"""
CURRENT DATA CONTEXT:
- Data available: {num_rows} rows x {num_cols} columns
- Sample data:
{safe_sample}
"""
            except Exception as e:
                data_context = f"\nData loading error: {str(e)}"
        else:
            data_context = "\nCURRENT DATA CONTEXT:\n- Data is available: NO"
        
        return f"""You are a specialized data analysis expert. Your role is to help analyze data using the available tools.

AVAILABLE TOOLS:
1. data_summary: Get comprehensive data overview (columns, types, statistics, sample)
2. generate_pandas_code: Generate pandas code from natural language query
3. execute_python_code: Execute pandas code safely on the DataFrame

IMPORTANT INSTRUCTIONS:
- You have access to a DataFrame with real data (see sample below)
- When user asks about data structure/columns/info use data_summary for complete overview
- When user wants specific analysis/filtering/calculations you can either:
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

IMPORTANT:
1. Do NOT include anything outside of CODE and an Explanation.
2. Explanation MUST be about what is understood from the QUESTION.
Your response must be exactly this format:
{{"code": "your_pandas_code_here", "explanation": "brief explanation"}}

{format_instructions}

CRITICAL REQUIREMENTS:
1. The code must assign the final result to a variable named 'result'
2. Use ONLY pandas operations - pandas and numpy are already imported as 'pd' and 'np'
3. UNDERSTAND what input wants exactly
4. The DataFrame is already loaded as 'df'
5. Do NOT include any import statements in your code

Question: {query}
Available Columns: {columns}
Data Types: {df_info}
"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["query", "columns", "df_info"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        # LLMChain yerine yeni format kullan
        chain = prompt | self.llm | self.parser
        
        logger.info(f"Generating code for query: {self.query}")
                        
        try:
            response = chain.invoke({
                "query": self.query,
                "columns": list(self.df.columns),
                "df_info": self.df.dtypes.to_dict()
            })
            
            # Handle different response formats
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
        """Setup data analysis tools"""
        logger.info("Setting up tools")
        
        def get_data_summary(query: str = "") -> str:
            """Get comprehensive data summary"""
            logger.info(f"Getting comprehensive data summary for query: {query}")

            if self.df is None:
                return "No data loaded"
            
            try:
                shape_info = f"{self.df.shape[0]} rows x {self.df.shape[1]} columns"
                columns_list = [str(col).replace('{', '[').replace('}', ']') for col in self.df.columns]
                columns_info = ", ".join(columns_list)
                
                return f"""
Data Summary:
- Shape: {shape_info}
- Columns: {columns_info}
- Data Types: {dict(self.df.dtypes)}
- Missing Values: {dict(self.df.isnull().sum())}
- Basic Statistics:
{self.df.describe().to_string()}
"""
            except Exception as e:
                logger.error(f"Error getting data summary: {e}")
                return f"Error getting data summary: {str(e)}"
        
        def generate_pandas_code(query: str = "") -> str: 
            """Generate pandas code here"""
            logger.info(f"Generate pandas code for query: {query}")

            try:
                if query:
                    self.query = query
                return self._generating_pandas_code()
            except Exception as e:
                self.log_error(f"Code generation failed: {e}")
                return f"Code generation failed: {str(e)}"

        def execute_python_code(pandas_code: str) -> str:
            """Execute pandas code safely on the DataFrame"""
            logger.info(f"Executing pandas code: {pandas_code}")
            
            if self.df is None:
                return "No data available for code execution"
            
            try:
                if not self._is_safe_code(pandas_code):
                    return "Code contains potentially dangerous operations and cannot be executed"
                
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
                
                # Get result
                result = local_vars.get("result", None)
                
                if result is not None:
                    logger.info("Code executed successfully")
                    
                    # Format result for display
                    if isinstance(result, pd.DataFrame):
                        if result.empty:
                            return "Code executed successfully. Result: Empty DataFrame"
                        # Limit output size for readability
                        if len(result) > 20:
                            display_result = result.head(20)
                            return f"Code executed successfully. Result (showing first 20 rows of {len(result)}):\n{display_result.to_string()}"
                        else:
                            return f"Code executed successfully. Result:\n{result.to_string()}"
                    
                    elif isinstance(result, pd.Series):
                        if len(result) > 20:
                            display_result = result.head(20)
                            return f"Code executed successfully. Result (showing first 20 items of {len(result)}):\n{display_result.to_string()}"
                        else:
                            return f"Code executed successfully. Result:\n{result.to_string()}"
                    
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
            Tool(
                name="data_summary", 
                description="Get data summary and statistics including shape, columns, data types, missing values, and basic statistics", 
                func=get_data_summary
            ),
            Tool(
                name="generate_pandas_code", 
                description="Generate pandas code from natural language query to analyze the DataFrame", 
                func=generate_pandas_code
            ),
            Tool(
                name="execute_python_code",
                description="Execute pandas code safely on the DataFrame. The code should assign results to a variable named 'result'",
                func=execute_python_code
            ),
        ]

    def process(self, query: str) -> AgentResult:
        """Process query through this specialized agent"""
        logger.info(f"Processing query: {query}")
        
        try:
            self.query = query
            
            if self.df is None:
                return AgentResult(
                    success=False,
                    error="No dataframe available",
                    metadata={"agent": self.agent_name, "query": query}
                )
            
            try:
                response = self.agent_executor.invoke({"input": query})
                logger.info(f"Agent response: {response}")
                
                return AgentResult(
                    success=True,
                    data=response,
                    metadata={"agent": self.agent_name, "query": query}
                )
            except Exception as agent_error:
                logger.warning(f"Agent failed: {agent_error}, using fallback")
                return AgentResult(success=False, error=error_msg)

            
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
                model_name="llama-3.3-70b-versatile",  
                api_key=groq_api_key,
                temperature=0.1
            )
        
            df = pd.read_csv("Data/goalscorers.csv")
            
            query = input("\nEnter your natural language query: ").strip()
            if query == "-1":
                break
            
            system = DataAnalysisAgent(llm=llm, df=df, query=query)
            
            # Process query
            result = system.process(query)
            
            if result.success:
                logger.info(f"**********{result.data}***********")
                print(f"Result: {result.data}")
            else:
                logger.error(f"**********{result.error}***********")
                print(f"Error: {result.error}")
            
        except Exception as e:
            logger.error(f"Application error: {e}")
            print(f"Application error: {e}")

if __name__ == "__main__":
    main()