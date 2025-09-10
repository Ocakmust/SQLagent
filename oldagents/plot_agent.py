import os
from typing import Optional
from langchain_groq import ChatGroq
import pandas as pd
import numpy as np
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.tools import Tool
from pydantic import BaseModel, Field
from utils.document import DocumentProcessor
from utils.loggerCenter import LoggerCenter
from utils.base_agent import AgentResult, BaseSpecializedAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import StructuredTool

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import glob

from utils.vectordeneme import ContextFind

logger = LoggerCenter().get_logger()

class PlotResult(BaseModel):
    success: bool
    plot_path: Optional[str] = None
    error_message: Optional[str] = None
    formatted_output: str = ""

class PlotOutput(BaseModel):
    code: str
    explanation: str
    plot_type: str

class VisualizationAgent(BaseSpecializedAgent):
    """Specialized agent for data visualization tasks"""
    
    def __init__(self, llm: ChatGroq, df: pd.DataFrame, doc_path: str = None, column_info_path: str = None, plots_dir: str = "plots"):
        self.df = df
        self.doc_path = doc_path
        self.column_info_path = column_info_path
        self.plots_dir = plots_dir
        self.plot_parser = PydanticOutputParser(pydantic_object=PlotOutput)

        os.makedirs(self.plots_dir, exist_ok=True)

        super().__init__("Visualization", llm)

        self.plot_result = None
        self.data_info = None
        self.data_info_loaded = False

        self.context_finder = None
        if self.doc_path is not None:
            try:
                self.context_finder = ContextFind(doc_path)
                logger.info("Context finder has been set")
            except Exception as e:
                logger.error(f"Error while starting context finder: {e}")
            
        self.columnInfo = None
        if self.column_info_path is not None:
            try:
                self.columnInfo = DocumentProcessor().extract_text_from_documents(column_info_path)
                logger.info("Column info uploaded")
            except Exception as e:
                logger.error(f"Error while starting column info: {e}")

        plt.style.use('default')
        sns.set_palette("husl")
        warnings.filterwarnings('ignore')

    def _get_system_prompt(self) -> str:
        """System prompt focused on visualization tasks"""
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
        
        return f"""You are a specialized data visualization expert. Your role is to help create beautiful and meaningful visualizations from data.

AVAILABLE TOOLS:
1. data_summary: Get comprehensive data overview (columns, statistics, sample, column infos, context/meaning of the columns)
2. generate_plot_code: Generate matplotlib/seaborn plotting code from natural language query
3. create_visualization: Create and save plots using matplotlib/seaborn

IMPORTANT INSTRUCTIONS and WORKFLOW:
- You have access to a DataFrame with real data (see sample below)
- When user asks about data structure/columns/info/context use data_summary for complete overview
- When user wants VISUALIZATIONS/plots/charts/graphs:
  * Use data_summary first to understand the data structure and content
  * Use generate_plot_code to generate appropriate plotting code
  * Use create_visualization to create and save the plot
- Always provide clear explanations of the visualizations created
- Suggest appropriate visualization types based on data types and user requirements

PLOTTING CAPABILITIES:
- Line plots, bar charts, histograms, scatter plots, box plots, heatmaps
- Distribution plots, correlation matrices, time series plots
- Subplots and multiple visualizations
- Custom styling with seaborn and matplotlib
- Interactive elements where appropriate

VISUALIZATION BEST PRACTICES:
1. Choose appropriate plot types based on data types:
   - Categorical data: bar plots, count plots, pie charts
   - Numerical data: histograms, box plots, scatter plots
   - Time series: line plots, area plots
   - Relationships: scatter plots, correlation heatmaps
2. Always include proper labels, titles, and legends
3. Use appropriate color schemes and styling
4. Handle missing values appropriately
5. Ensure plots are readable and informative

ANTI-HALLUCINATION RULES (CRITICAL):
1. NEVER make up column names or data not present in the actual DataFrame
2. NEVER create fictional explanations about data patterns
3. ALWAYS base visualizations on actual data structure and content
4. If unsure about data structure, always use data_summary first

VERY IMPORTANT:
- Focus ONLY on visualization tasks
- Always use appropriate figure sizes for clarity
- Save all plots with high quality (dpi=300)
- Provide meaningful insights about what the visualization shows
- NEVER IMPORT ANYTHING IN YOUR CODE.

{data_context}

Your goal is to help users create compelling and informative visualizations from their data!
"""

    def _generating_plot_code(self) -> str:
        """Generate matplotlib/seaborn plotting code for the given query"""
        template = """
You are an expert data visualization specialist. Generate Python plotting code using matplotlib and seaborn for the given request.

YOUR DATA CONTEXT:
{data_info}

IMPORTANT REQUIREMENTS:
1. Generate clean, well-structured plotting code
2. Use appropriate plot types based on data types and user request
3. Always include proper labels, titles, and styling
4. Handle missing values and edge cases
5. Use professional color schemes and layouts
6. Save the plot with high quality

CRITICAL IMPORT RULES:
- DO NOT include ANY import statements in your code
- All necessary libraries are already imported and available:
  * matplotlib.pyplot as 'plt'
  * seaborn as 'sns' 
  * pandas as 'pd'
  * numpy as 'np'
- The DataFrame is already loaded as 'df'

Your response must be exactly this format:
{{"code": "your_plotting_code_here", "explanation": "brief explanation of the visualization", "plot_type": "type_of_plot"}}

{format_instructions}

BE CAREFUL -> Return only valid JSON. Do not use triple quotes or code blocks. Otherwise you will face error.
VERY IMPORTANT -> ALWAYS END YOUR CODE SAVING THE FIGURE WITH plt.savefig('{plots_dir}/plot_{short_query}.png', dpi=300, bbox_inches='tight'

CRITICAL REQUIREMENTS:
1. matplotlib.pyplot is imported as 'plt', seaborn is imported as 'sns'
2. The DataFrame is already loaded as 'df'
3. Use figure size plt.figure(figsize=(12, 8)) or similar for better visualization
4. Always end with plt.savefig('{plots_dir}/plot_{short_query}.png', dpi=300, bbox_inches='tight')
5. Include plt.tight_layout() for better spacing
6. Do NOT include ANY import statements - they are already available
7. Handle missing values with appropriate methods
8. Use clear, descriptive titles and axis labels

CODE EXAMPLE FORMAT (NO IMPORTS):
# Clean plotting code without imports
plt.figure(figsize=(12, 8))
# Your plotting code here
plt.title('Your Title')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.tight_layout()
plt.savefig('{plots_dir}/plot_{short_query}.png', dpi=300, bbox_inches='tight')
plt.show()

AVAILABLE PLOT TYPES AND WHEN TO USE:
- Bar plots: For categorical data comparison
- Line plots: For time series or continuous data trends  
- Histograms: For distribution of numerical data
- Scatter plots: For relationships between numerical variables
- Box plots: For distribution comparison across categories
- Heatmaps: For correlation matrices or 2D data
- Violin plots: For detailed distribution shapes
- Pair plots: For multiple variable relationships
- Count plots: For frequency of categorical values
- Subplots: For multiple related visualizations

Question: {query}
"""
        
        words = self.query.strip().split()
        first_three = words[:3]
        short_query = "".join(first_three)

        prompt = PromptTemplate(
            template=template,
            input_variables=["query", "data_info", "plots_dir","short_query"],
            partial_variables={"format_instructions": self.plot_parser.get_format_instructions()}
        )

        chain = prompt | self.llm | self.plot_parser
        
        logger.info(f"Generating plot code for query: {self.query}")
                        
        try:
            response = chain.invoke({
                "query": self.query, 
                "data_info": self.data_info,
                "plots_dir": self.plots_dir,
                "short_query": short_query
            })
            
            if isinstance(response, dict):
                code = response.get("code", str(response))
            else:
                code = response.code if hasattr(response, 'code') else str(response)
            
            code = self._clean_imports_from_code(code)
                        
            logger.info(f"Plot code generated: {code}")
            return code
            
        except Exception as e:
            logger.error(f"Error in plot code generation: {e}")
            return f"Error generating plot code: {str(e)}"
    
    def _clean_imports_from_code(self, code: str) -> str:
        """Remove import statements from generated code"""
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            if (line_stripped.startswith('import ') or 
                line_stripped.startswith('from ') or
                line_stripped.startswith('# import') or
                line_stripped.startswith('# from')):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _setup_tools(self):
        logger.info("Setting up visualization tools")

        class PlotInfoInput(BaseModel):
            query: Optional[str] = Field(default="", description="Optional user query to get relevant context and data info")
        
        class PlotGenerationInput(BaseModel):
            query: str = Field(description="User request for plot code generation")
        
        class PlotExecutionInput(BaseModel):
            plot_code: str = Field(description="Plot code to execute")
        
        def get_data_summary(query: str = "") -> str:
            """Get comprehensive data summary for visualization planning"""
            logger.info(f"Getting data summary for visualization planning: {query}")

            if self.df is None:
                return "No data loaded"
            
            context_info = "No additional context available"
            columnInfo_info = "No column information available"
            doc_link = "No source available"
            columnInfo_link = "No source available"
            
            try:
                shape_info = f"{self.df.shape[0]} rows x {self.df.shape[1]} columns"
                columns_list = [str(col).replace('{', '[').replace('}', ']') for col in self.df.columns]
                columns = ", ".join(columns_list)
                
                d_types = dict(self.df.dtypes)
                
                numeric_cols = self.df.select_dtypes(include=['number']).columns
                categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
                
                stats_info = ""
                if len(numeric_cols) > 0:
                    stats_info += f"\nNumerical columns: {list(numeric_cols)}"
                if len(categorical_cols) > 0:
                    stats_info += f"\nCategorical columns: {list(categorical_cols)}"

                sample_data = self.df.head(3).to_string()

                if self.context_finder and query:
                    try:
                        context_info = self.context_finder.return_context(query, top_k=3)
                        if not context_info.strip():
                            context_info = "No relevant context found"
                        doc_link = self.doc_path
                    except Exception as e:
                        logger.error(f"Error getting context: {e}")

                if self.columnInfo:
                    try:
                        columnInfo_info = self.columnInfo
                        columnInfo_link = self.column_info_path
                    except Exception as e:
                        logger.error(f"Error getting column info: {e}")

                self.data_info = f"""
Data Summary for Visualization:
- Shape: {shape_info}
- Columns: {columns}
- Data Types: {d_types}{stats_info}

Sample Data:
{sample_data}

Relevant Context:
{context_info}

Column Information:
{columnInfo_info}

Source of context: {doc_link}
Source of column info: {columnInfo_link}
"""
                self.data_info_loaded = True
                return self.data_info
            
            except Exception as e:
                logger.error(f"Error getting data summary: {e}")
                return f"Error getting data summary: {str(e)}"

        def generate_plot_code(query: str = "") -> str:
            """Generate matplotlib/seaborn plotting code"""
            logger.info(f"Generate plot code for query: {query}")

            if not self.data_info_loaded or not self.data_info:
                logger.info("Data info not loaded, loading automatically...")
                get_data_summary(query)

            try:
                if query:
                    self.query = query
                return self._generating_plot_code()
            except Exception as e:
                logger.error(f"Plot code generation failed: {e}")
                return f"Plot code generation failed: {str(e)}"

        def create_visualization(plot_code: str) -> str:
            """Create and save visualizations using matplotlib/seaborn"""
            logger.info(f"Creating visualization with code: {plot_code}")
            
            if self.df is None:
                return "No data available for visualization"
            
            try:
                plt.clf()
                plt.close('all')
                
                clean_code = self._clean_imports_from_code(plot_code)
                
                safe_globals = {
                    'pd': pd,
                    'np': np,
                    'df': self.df.copy(),
                    'plt': plt,
                    'sns': sns,
                    'os': os,
                    'warnings': warnings,
                    'matplotlib': __import__('matplotlib'),
                    'seaborn': sns,
                    '__builtins__': {
                        'len': len, 'str': str, 'int': int, 'float': float,
                        'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                        'range': range, 'enumerate': enumerate, 'zip': zip,
                        'sorted': sorted, 'reversed': reversed, 'sum': sum,
                        'min': min, 'max': max, 'abs': abs, 'round': round,
                        'print': print, 'isinstance': isinstance, 'hasattr': hasattr
                    }
                }
                
                local_vars = {}
                
                exec(clean_code, safe_globals, local_vars)
                
                plot_files = (glob.glob("*.png") + glob.glob("*.jpg") + 
                            glob.glob("*.jpeg") + glob.glob(f"{self.plots_dir}/*.png") +
                            glob.glob(f"{self.plots_dir}/*.jpg") + glob.glob(f"{self.plots_dir}/*.jpeg"))
                
                if plot_files:
                    plot_path = max(plot_files, key=os.path.getctime)
                    
                    self.plot_result = PlotResult(
                        success=True,
                        plot_path=plot_path,
                        formatted_output=f"Visualization created and saved to: {plot_path}"
                    )
                    
                    plt.close('all')  
                    return f"Visualization created successfully and saved to: {plot_path}"
                else:
                    return "Visualization code executed but no plot file was found. Make sure to use plt.savefig() in your code."
                
            except Exception as e:
                error_msg = f"Visualization creation failed: {str(e)}"
                logger.error(error_msg)
                
                self.plot_result = PlotResult(
                    success=False,
                    error_message=error_msg,
                    formatted_output=error_msg
                )
                
                plt.close('all')  
                return error_msg
                
        self.tools = [
            StructuredTool.from_function(
                name="data_summary", 
                description="Get comprehensive data summary including shape, columns, data types, and sample data for visualization planning", 
                func=get_data_summary,
                args_schema=PlotInfoInput
            ),
            StructuredTool.from_function(
                name="generate_plot_code",
                description="Generate matplotlib/seaborn plotting code from natural language query for data visualization",
                func=generate_plot_code,
                args_schema=PlotGenerationInput
            ),
            StructuredTool.from_function(
                name="create_visualization",
                description="Create and save plots/charts using matplotlib and seaborn. Provide the plotting code as input.",
                func=create_visualization,
                args_schema=PlotExecutionInput
            ),
        ]

    def _clean_imports_from_code(self, code: str) -> str:
        """Remove import statements from generated code"""
        if not isinstance(code, str):
            return str(code)
            
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            if (line_stripped.startswith('import ') or 
                line_stripped.startswith('from ') or
                line_stripped.startswith('# import') or
                line_stripped.startswith('# from') or
                ('import ' in line_stripped and line_stripped.startswith('#'))):
                logger.info(f"Removing import line: {line}")
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def process(self, query: str) -> AgentResult:
        """Process visualization query through this specialized agent"""
        logger.info(f"Processing visualization query: {query}")
        
        try:
            self.query = query

            response = self.agent_executor.invoke({"input": query})
            
            metadata = {
                "agent": self.agent_name, 
                "query": query
            }
            
            if self.plot_result:
                metadata["plot_path"] = self.plot_result.plot_path
                metadata["plot_success"] = self.plot_result.success
            
            return AgentResult(
                success=True,
                data=response,
                metadata=metadata
            )
        
        except Exception as e:
            error_msg = f"Error processing visualization query '{query}': {str(e)}"
            logger.error(error_msg)
            return AgentResult(success=False, error=error_msg)

    
def main():
    """Test the visualization agent"""
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
            
            query = input("\nEnter your visualization request: ").strip()
            if query == "-1":
                break
            
            viz_agent = VisualizationAgent(llm=llm, df=df, plots_dir="plots")
            
            result = viz_agent.process(query)
            result_info = result.data["output"]
                
            if result.metadata.get("plot_path"):
                plot_path = result.metadata["plot_path"]
                print(f"Plot saved to: {plot_path}")
            
            if result.success:
                print(f"Result: {result_info}\n")
                print("=" * 50)
            else:
                print(f"Error: {result.error}")
            
        except Exception as e:
            logger.error(f"Application error: {e}")
            print(f"Application error: {e}")

if __name__ == "__main__":
    main()