import os
import pandas as pd
from typing import Optional, Dict, Any, TypedDict, Literal
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from utils.document import DocumentProcessor

from utils.loggerCenter import LoggerCenter
from utils.base_agent import AgentResult, BaseSpecializedAgent
from oldagents.pandas_agent import DataAnalysisAgent
from oldagents.db_agent import SQLQuerryAgent
from api_agent import ExternalAPIAgent
from utils.db_dao import DatabaseManager
from utils.vectordeneme import ContextFind

logger = LoggerCenter().get_logger()

class RoutingDecision(BaseModel):
    agent: Literal["data", "sql", "api"] = Field(description="Selected agent to handle the query")
    reasoning: str = Field(description="Brief explanation of why this agent was selected")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")

class RouterState(TypedDict):
    query: str
    route_decision: Optional[str]
    routing_reasoning: Optional[str]
    confidence_score: Optional[float]
    result: Optional[str]
    error: Optional[str]
    metadata: Optional[Dict[str, Any]]
    csv_dataframe: Optional[str] 
    sql_dataframe: Optional[str]

class LangGraphIntelligentRouter:
    def __init__(self, llm: ChatGroq, config: Dict[str, Any]):
        self.config = config
        self.llm = llm
        self.sub_agents = {}
        self.dataframe = None
        self.db_manager = None
        self.context_finder = None
        self.column_info = None
        
        self._load_resources()
        self._initialize_agents()
        self.app = self._build_graph()
    
    def _load_resources(self):
        csv_path = self.config.get('csv_path')
        if csv_path and Path(csv_path).exists():
            try:
                self.dataframe = pd.read_csv(csv_path)
                logger.info(f"CSV loaded: {self.dataframe.shape}")
            except Exception as e:
                logger.error(f"CSV loading failed: {e}")
        
        if self.config.get('db_params'):
            try:
                self.db_manager = DatabaseManager(self.config['db_params'])
                self.db_manager.connect()
                logger.info("Database connected")
            except Exception as e:
                logger.error(f"Database error: {e}")
        
        doc_path = self.config.get('doc_path')
        if doc_path and Path(doc_path).exists():
            try:
                self.context_finder = ContextFind(doc_path)
                logger.info("PDF context ready")
            except Exception as e:
                logger.error(f"PDF context error: {e}")
        
        column_info_path = self.config.get('column_info')
        if column_info_path and Path(column_info_path).exists():
            try:
                doc_processor = DocumentProcessor()
                self.column_info = doc_processor.extract_text_from_documents(column_info_path)
                logger.info("Column info loaded")
            except Exception as e:
                logger.error(f"Column info loading failed: {e}")
    
    def _initialize_agents(self):
        try:
            if self.dataframe is not None:
                self.sub_agents['data'] = DataAnalysisAgent(
                    llm=self.llm,
                    df=self.dataframe,
                    doc_path=self.config.get('doc_path'),
                    column_info_path=self.config.get('column_info')
                )
                logger.info("Data Analysis Agent ready")
            
            if self.db_manager is not None:
                self.sub_agents['sql'] = SQLQuerryAgent(
                    llm=self.llm,
                    db_manager=self.db_manager,
                    doc_path=self.config.get('doc_path'),
                    columnInfo_path=self.config.get('column_info')
                )
                logger.info("SQL Agent ready")
            
            self.sub_agents['api'] = ExternalAPIAgent(llm=self.llm)
            logger.info("API Agent ready")
        except Exception as e:
            logger.error(f"Agent initialization error: {e}")

    def _get_detailed_resources_summary(self) -> str:
        resources = []
        if self.dataframe is not None:
            columns = list(self.dataframe.columns)
            sample_data = ""
            if not self.dataframe.empty:
                sample_data = f"\nSample data preview:\n{self.dataframe.head(3).to_string()}"
            
            resources.append(f"""CSV DATA SOURCE:
- File contains {self.dataframe.shape[0]} rows and {self.dataframe.shape[1]} columns
- Available columns: {', '.join(columns[:10])}{'...' if len(columns) > 10 else ''}
- Data types: {dict(self.dataframe.dtypes.head(10))}
{sample_data}
- Best for: Statistical analysis, data exploration, charts, pandas operations""")
        
        if self.db_manager:
            try:
                tables = self.db_manager.get_table_names()
                if tables:
                    schema_info = []
                    for table in tables[:3]:
                        try:
                            schema = self.db_manager.get_table_schema(table)
                            columns_info = ', '.join([f"{col['column_name']} ({col['data_type']})" 
                                                    for col in schema[:5]])
                            schema_info.append(f"  - {table}: {columns_info}")
                        except Exception:
                            schema_info.append(f"  - {table}: Schema unavailable")
                    
                    resources.append(f"""DATABASE SOURCE:
- Connected to PostgreSQL database
- Available tables ({len(tables)} total): {', '.join(tables[:5])}{'...' if len(tables) > 5 else ''}
- Table schemas:
{chr(10).join(schema_info)}
- Best for: Complex queries, joins, structured data operations, SQL analysis""")
                else:
                    resources.append("DATABASE: Connected but no accessible tables found")
            except Exception:
                resources.append("DATABASE: Connected but schema unavailable")
        
        if self.column_info:
            resources.append("""COLUMN DOCUMENTATION:
- Detailed column descriptions and metadata available
- Best for: Understanding data structure and meaning""")
        
        resources.append("""EXTERNAL API SOURCE:
- Weather information and forecasts
- Current news and events
- Real-time external data
- Best for: Current information, weather, news, live data""")
        
        return "\n\n".join(resources) if resources else "No resources available"

    def _llm_routing_node(self, state: RouterState) -> RouterState:
        query = state["query"]
        logger.info(f"Making intelligent LLM-based routing decision for: {query}")
        
        available_agents = []
        if 'data' in self.sub_agents:
            available_agents.append("data")
        if 'sql' in self.sub_agents:
            available_agents.append("sql")
        if 'api' in self.sub_agents:
            available_agents.append("api")
        
        parser = PydanticOutputParser(pydantic_object=RoutingDecision)
        
        routing_prompt = f"""You are an intelligent routing system that analyzes user queries and selects the most appropriate specialized agent.

USER QUERY:
"{query}"

RESOURCES AND AGENTS:
{self._get_detailed_resources_summary()}

AGENT CAPABILITIES:
• DATA AGENT: Handles CSV file analysis, statistical computations, data visualization, pandas operations, exploratory data analysis
• SQL AGENT: Executes database queries, joins, aggregations, database information retrieval
• API AGENT: Fetches real-time external information like weather, news, events, live data

{parser.get_format_instructions()}"""

        try:
            response = self.llm.invoke(routing_prompt)
            routing_decision = parser.parse(response.content)
            
            if routing_decision.agent not in available_agents:
                logger.warning(f"Invalid route '{routing_decision.agent}', using fallback")
                if 'data' in available_agents:
                    route_decision = "data"
                    reasoning = "Fallback: Selected data agent"
                    confidence = 0.7
                elif 'sql' in available_agents:
                    route_decision = "sql"
                    reasoning = "Fallback: Selected SQL agent"
                    confidence = 0.6
                else:
                    route_decision = "api"
                    reasoning = "Fallback: Selected API agent"
                    confidence = 0.5
            else:
                route_decision = routing_decision.agent
                reasoning = routing_decision.reasoning
                confidence = routing_decision.confidence
            
            return {
                **state,
                "route_decision": route_decision,
                "routing_reasoning": reasoning,
                "confidence_score": confidence
            }
        except Exception as e:
            logger.error(f"LLM routing error: {e}")
            if 'data' in available_agents and self.dataframe is not None:
                route_decision = "data"
                reasoning = "Error fallback: CSV data available"
            elif 'sql' in available_agents and self.db_manager is not None:
                route_decision = "sql"
                reasoning = "Error fallback: Database available"
            else:
                route_decision = "api"
                reasoning = "Error fallback: API agent"
            
            return {
                **state,
                "route_decision": route_decision,
                "routing_reasoning": reasoning,
                "confidence_score": 0.4
            }

    def _data_agent_node(self, state: RouterState) -> RouterState:
        try:
            logger.info("Executing Data Analysis Agent")
            result = self.sub_agents['data'].process(state["query"])
            
            csv_dataframe = None
            if result.success and hasattr(result, 'metadata') and result.metadata:
                csv_dataframe = result.metadata.get('dataframe')
            
            if result.success:
                output = result.data.get('output', result.data) if isinstance(result.data, dict) else result.data
                return {
                    **state,
                    "result": str(output),
                    "csv_dataframe": csv_dataframe.to_json(orient='records'),
                }
            else:
                return {**state, "error": f"Data analysis error: {result.error}"}
        except Exception as e:
            logger.error(f"Data agent error: {e}")
            return {**state, "error": f"Data agent execution error: {str(e)}"}

    def _sql_agent_node(self, state: RouterState) -> RouterState:
        try:
            logger.info("Executing SQL Agent")
            result = self.sub_agents['sql'].process(state["query"])
            
            sql_dataframe = None
            if result.success and hasattr(result, 'metadata') and result.metadata:
                sql_dataframe = result.metadata.get('dataframe')
            
            if result.success:
                output = result.data.get('output', result.data) if isinstance(result.data, dict) else result.data
                return {
                    **state,
                    "result": str(output),
                    "sql_dataframe": sql_dataframe.to_json(orient='records'),
                }
            else:
                return {**state, "error": f"SQL error: {result.error}"}
        except Exception as e:
            logger.error(f"SQL agent error: {e}")
            return {**state, "error": f"SQL agent execution error: {str(e)}"}

    def _api_agent_node(self, state: RouterState) -> RouterState:
        try:
            logger.info("Executing API Agent")
            result = self.sub_agents['api'].process(state["query"])
            
            if result.success:
                return {
                    **state,
                    "result": str(result.data),
                    "metadata": result.metadata if hasattr(result, 'metadata') else {}
                }
            else:
                return {**state, "error": f"API error: {result.error}"}
        except Exception as e:
            logger.error(f"API agent error: {e}")
            return {**state, "error": f"API agent execution error: {str(e)}"}

    def _route_condition(self, state: RouterState) -> Literal["data_agent", "sql_agent", "api_agent"]:
        route = state.get("route_decision")
        route_map = {"data": "data_agent", "sql": "sql_agent", "api": "api_agent"}
        return route_map.get(route, "api_agent")

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(RouterState)
        workflow.add_node("llm_routing", self._llm_routing_node)
        workflow.add_node("data_agent", self._data_agent_node)
        workflow.add_node("sql_agent", self._sql_agent_node)
        workflow.add_node("api_agent", self._api_agent_node)
        
        workflow.set_entry_point("llm_routing")
        workflow.add_conditional_edges("llm_routing", self._route_condition, {
            "data_agent": "data_agent",
            "sql_agent": "sql_agent",
            "api_agent": "api_agent"
        })
        workflow.add_edge("data_agent", END)
        workflow.add_edge("sql_agent", END)
        workflow.add_edge("api_agent", END)
        
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def process(self, query: str) -> AgentResult:
        logger.info(f"LangGraph Intelligent Router processing: {query}")
        try:
            initial_state = RouterState(
                query=query,
                route_decision=None,
                routing_reasoning=None,
                confidence_score=None,
                result=None,
                error=None,
                metadata=None,
                csv_dataframe=None,
                sql_dataframe=None
            )
            
            config = {"configurable": {"thread_id": f"router_{hash(query) % 10000}"}}
            final_state = self.app.invoke(initial_state, config)
            
            if final_state.get("error"):
                return AgentResult(success=False, error=final_state["error"])
            

            csv_dataframe=None
            sql_dataframe=None
            if final_state.get("csv_dataframe"):
                try:
                    import json
                    csv_data = json.loads(final_state["csv_dataframe"])
                    csv_dataframe = pd.DataFrame(csv_data)
                except Exception as e:
                    logger.warning(f"CSV DataFrame conversion failed: {e}")
            
            if final_state.get("sql_dataframe"):
                try:
                    import json
                    sql_data = json.loads(final_state["sql_dataframe"])
                    sql_dataframe = pd.DataFrame(sql_data)
                except Exception as e:
                    logger.warning(f"SQL DataFrame conversion failed: {e}")
                
            return AgentResult(
                success=True,
                data={"output": final_state.get("result", "No result generated")},
                metadata={
                    "agent": "LangGraphIntelligentRouter",
                    "route_decision": final_state.get("route_decision"),
                    "routing_reasoning": final_state.get("routing_reasoning"),
                    "confidence_score": final_state.get("confidence_score"),
                    "original_query": query,
                    "sql_dataframe": csv_dataframe,
                    "csv_dataframe": sql_dataframe
                }
            )
        except Exception as e:
            error_msg = f"LangGraph Intelligent Router error: {str(e)}"
            logger.error(error_msg)
            return AgentResult(success=False, error=error_msg)


def main():
    print("LangGraph Intelligent Router System")
    print("=" * 50)
    
    config = {
        'db_params': {
            "host": "localhost",
            "database": "musteri_db", 
            "user": "postgres",
            "password": "123",
            "port": "5432"
        },
        'csv_path': "Data/goalscorers.csv",
        "doc_path":"temp_columns.pdf"
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
        
        print("Initializing Intelligent LangGraph router...")
        router = LangGraphIntelligentRouter(llm=llm, config=config)
        print("Router ready!\n")
        

        test_queries = [
            "Kadın girişimci olup bankaya sadık olan en risksiz 5 kişiyi döndür."
        ]
        
        for query in test_queries:
            print(f"Processing: {query}")
            result = router.process(query)
            
            if result.success:
                print(f"Result: {result.data.get('output', 'No output')}")
                print(f"Route: {result.metadata.get('route_decision')}")
                print(f"Reasoning: {result.metadata.get('routing_reasoning')}")
                print(f"Confidence: {result.metadata.get('confidence_score', 'N/A')}")
                
                if result.metadata.get("sql_dataframe") is not None:
                    sql_df = result.metadata["sql_dataframe"] 
                    if isinstance(sql_df, pd.DataFrame) and not sql_df.empty:
                        print(f"SQL DataFrame: {sql_df.shape}")
                
                if result.metadata.get("csv_dataframe") is not None:
                    csv_df = result.metadata["csv_dataframe"]
                    if isinstance(csv_df, pd.DataFrame) and not csv_df.empty:
                        print(f"CSV DataFrame: {csv_df.shape}")
            else:
                print(f"Error: {result.error}")
            
            print("\n" + "-" * 50 + "\n")
    except Exception as e:
        print(f"System initialization failed: {e}")

if __name__ == "__main__":
    main()
