import os
import pandas as pd
from typing import Optional, Dict, Any, List
from pathlib import Path
from langchain_groq import ChatGroq
from langchain.tools import Tool
from loggerCenter import LoggerCenter

# Import agents and utilities
from utils import AgentResult, BaseSpecializedAgent, DatabaseManager, ContextFind
from analiz import DataAnalysisAgent
from db_agent import SQLQuerryAgent
from api_agent import ExternalAPIAgent

logger = LoggerCenter().get_logger()

class RouterAgent(BaseSpecializedAgent):
    """Smart router agent that uses tools to determine and execute the best approach for user queries"""
    
    def __init__(self, llm: ChatGroq, config: Dict[str, Any]):
        """
        Initialize router agent with configuration
        
        config should contain:
        - db_params: Database connection parameters (optional)
        - pdf_path: Path to PDF for context (optional)
        - csv_path: Path to CSV data file (optional)
        """
        self.config = config
        
        # Initialize resources based on config
        self.db_manager = self._setup_database() if config.get('db_params') else None
        self.pdf_path = config.get('pdf_path')
        self.csv_path = config.get('csv_path')
        self.dataframe = self._load_dataframe() if self.csv_path else None
        self.context_finder = self._setup_context_finder() if self.pdf_path else None
        
        # Initialize sub-agents
        self.sub_agents = {}
        self._initialize_sub_agents(llm)
        
        # Initialize as BaseSpecializedAgent
        super().__init__("RouterAgent", llm)
    
    def _setup_database(self) -> Optional[DatabaseManager]:
        """Setup database manager if config provided"""
        try:
            db_params = self.config['db_params']
            db_manager = DatabaseManager(db_params)
            db_manager.connect()
            logger.info("Database connection established")
            return db_manager
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return None
    
    def _load_dataframe(self) -> Optional[pd.DataFrame]:
        """Load CSV data if path provided"""
        try:
            if not self.csv_path or not Path(self.csv_path).exists():
                return None
            df = pd.read_csv(self.csv_path)
            logger.info(f"CSV data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"CSV loading failed: {e}")
            return None
    
    def _setup_context_finder(self) -> Optional[ContextFind]:
        """Setup context finder for PDF"""
        try:
            if not self.pdf_path or not Path(self.pdf_path).exists():
                return None
            return ContextFind(self.pdf_path)
        except Exception as e:
            logger.error(f"Context finder setup failed: {e}")
            return None
    
    def _initialize_sub_agents(self, llm: ChatGroq):
        """Initialize sub-agents based on available resources"""
        try:
            # Data Analysis Agent
            if self.dataframe is not None:
                self.sub_agents['data_analysis'] = DataAnalysisAgent(
                    llm=llm, df=self.dataframe, query=""
                )
                logger.info("Data Analysis Agent initialized")
            
            # SQL Agent
            if self.db_manager and self.pdf_path:
                self.sub_agents['sql_database'] = SQLQuerryAgent(
                    llm=llm, pdf_path=self.pdf_path, query="", db_manager=self.db_manager
                )
                logger.info("SQL Query Agent initialized")
            
            # External API Agent
            self.sub_agents['external_api'] = ExternalAPIAgent(llm=llm)
            logger.info("External API Agent initialized")
            
        except Exception as e:
            logger.error(f"Sub-agent initialization failed: {e}")

    def _get_system_prompt(self) -> str:
        """System prompt for the router agent"""
        # Build context about available resources
        resources_info = self._get_resources_context()
        
        return f"""You are an intelligent router agent that determines the best approach to answer user queries.

AVAILABLE TOOLS:
1. check_available_resources: Check what data sources and capabilities are available
2. analyze_query_context: Get relevant context from PDF documentation for understanding query intent
3. route_to_data_analysis: Route to data analysis agent for CSV data operations
4. route_to_sql_database: Route to SQL database agent for database queries with business context
5. route_to_external_api: Route to external API agent for weather, news, and external data

CURRENT SYSTEM RESOURCES:
{resources_info}

ROUTING STRATEGY:
- First, check available resources to understand capabilities
- If query involves data analysis, statistics, CSV operations → use route_to_data_analysis
- If query involves database, SQL, business rules from PDF → use route_to_sql_database
- If query involves weather, news, external APIs → use route_to_external_api
- Use analyze_query_context to understand business intent when needed

INSTRUCTIONS:
- Always check resources first to know what's available
- Choose the most appropriate tool based on query content and available resources
- Provide clear explanations of routing decisions
- Handle cases where requested resource is not available gracefully

Your goal is to intelligently route queries to the best available agent!"""

    def _get_resources_context(self) -> str:
        """Get context about available resources"""
        resources = []
        
        if self.dataframe is not None:
            resources.append(f"✅ CSV Data: {self.dataframe.shape[0]} rows, {self.dataframe.shape[1]} columns")
        else:
            resources.append("❌ CSV Data: Not available")
        
        if self.db_manager and self.db_manager.is_connected():
            try:
                tables = self.db_manager.get_table_names()
                resources.append(f"✅ Database: Connected, {len(tables)} tables available")
            except:
                resources.append("⚠️ Database: Connected but schema unavailable")
        else:
            resources.append("❌ Database: Not available")
        
        if self.context_finder:
            resources.append(f"✅ PDF Context: Available ({self.pdf_path})")
        else:
            resources.append("❌ PDF Context: Not available")
        
        resources.append("✅ External APIs: Weather and News available")
        
        return "\n".join(resources)

    def _setup_tools(self):
        """Setup router tools"""
        
        def check_available_resources() -> str:
            """Check what data sources and capabilities are currently available"""
            try:
                status = {
                    "csv_data": {
                        "available": self.dataframe is not None,
                        "details": f"{self.dataframe.shape[0]} rows, {self.dataframe.shape[1]} columns" if self.dataframe is not None else "Not loaded"
                    },
                    "database": {
                        "available": self.db_manager is not None and self.db_manager.is_connected(),
                        "details": f"{len(self.db_manager.get_table_names())} tables" if self.db_manager and self.db_manager.is_connected() else "Not connected"
                    },
                    "pdf_context": {
                        "available": self.context_finder is not None,
                        "details": f"Document: {Path(self.pdf_path).name}" if self.pdf_path else "Not available"
                    },
                    "external_apis": {
                        "available": True,
                        "details": "Weather and News APIs available"
                    },
                    "available_agents": list(self.sub_agents.keys())
                }
                
                result = "SYSTEM RESOURCES STATUS:\n\n"
                for resource, info in status.items():
                    if resource == "available_agents":
                        result += f"Available Agents: {', '.join(info)}\n"
                    else:
                        status_icon = "✅" if info["available"] else "❌"
                        result += f"{status_icon} {resource.replace('_', ' ').title()}: {info['details']}\n"
                
                return result
                
            except Exception as e:
                return f"Error checking resources: {str(e)}"

        def analyze_query_context(user_query: str) -> str:
            """Analyze query context using PDF documentation to understand business intent"""
            try:
                if not self.context_finder:
                    return "PDF context not available for query analysis"
                
                # Get relevant context from PDF
                context = self.context_finder.return_context(user_query, top_k=3)
                
                if context:
                    return f"""QUERY CONTEXT ANALYSIS:

User Query: {user_query}

Relevant Business Context from PDF:
{context}

This context suggests the query relates to business rules and may benefit from database operations."""
                else:
                    return f"No relevant context found in PDF for query: {user_query}"
                    
            except Exception as e:
                return f"Error analyzing query context: {str(e)}"

        def route_to_data_analysis(user_query: str) -> str:
            """Route query to data analysis agent for CSV data operations"""
            try:
                if 'data_analysis' not in self.sub_agents:
                    return "❌ Data Analysis Agent not available. CSV data not loaded."
                
                logger.info(f"Routing to Data Analysis Agent: {user_query}")
                agent = self.sub_agents['data_analysis']
                result = agent.process(user_query)
                
                if result.success:
                    return f"✅ Data Analysis Result:\n{result.data.get('output', result.data)}"
                else:
                    return f"❌ Data Analysis Error: {result.error}"
                    
            except Exception as e:
                return f"❌ Error in data analysis routing: {str(e)}"

        def route_to_sql_database(user_query: str) -> str:
            """Route query to SQL database agent for database queries with business context"""
            try:
                if 'sql_database' not in self.sub_agents:
                    return "❌ SQL Database Agent not available. Database or PDF context not available."
                
                logger.info(f"Routing to SQL Database Agent: {user_query}")
                agent = self.sub_agents['sql_database']
                result = agent.process(user_query)
                
                if result.success:
                    return f"✅ SQL Database Result:\n{result.data.get('output', result.data)}"
                else:
                    return f"❌ SQL Database Error: {result.error}"
                    
            except Exception as e:
                return f"❌ Error in SQL database routing: {str(e)}"

        def route_to_external_api(user_query: str) -> str:
            """Route query to external API agent for weather, news, and external data"""
            try:
                if 'external_api' not in self.sub_agents:
                    return "❌ External API Agent not available."
                
                logger.info(f"Routing to External API Agent: {user_query}")
                agent = self.sub_agents['external_api']
                result = agent.process(user_query)
                
                if result.success:
                    return f"✅ External API Result:\n{result.data}"
                else:
                    return f"❌ External API Error: {result.error}"
                    
            except Exception as e:
                return f"❌ Error in external API routing: {str(e)}"

        self.tools = [
            Tool(
                name="check_available_resources",
                description="Check what data sources and capabilities are currently available in the system",
                func=check_available_resources
            ),
            Tool(
                name="analyze_query_context",
                description="Analyze user query context using PDF documentation to understand business intent and requirements",
                func=analyze_query_context
            ),
            Tool(
                name="route_to_data_analysis",
                description="Route query to data analysis agent for CSV data operations, statistics, and data manipulation",
                func=route_to_data_analysis
            ),
            Tool(
                name="route_to_sql_database",
                description="Route query to SQL database agent for database queries with business context from PDF",
                func=route_to_sql_database
            ),
            Tool(
                name="route_to_external_api",
                description="Route query to external API agent for weather information, news articles, and external data",
                func=route_to_external_api
            ),
        ]

    def process(self, query: str) -> AgentResult:
        """Process user query through intelligent routing"""
        logger.info(f"Router processing query: {query}")
        
        try:
            response = self.agent_executor.invoke({"input": query})
            
            return AgentResult(
                success=True,
                data=response,
                metadata={
                    "agent": self.agent_name,
                    "query": query,
                    "available_resources": {
                        "csv_data": self.dataframe is not None,
                        "database": self.db_manager is not None,
                        "pdf_context": self.context_finder is not None,
                        "external_api": True
                    }
                }
            )
            
        except Exception as e:
            error_msg = f"Router processing failed: {str(e)}"
            logger.error(error_msg)
            return AgentResult(success=False, error=error_msg)

def main():
    """Main application with router agent"""
    print("🤖 Intelligent Multi-Agent Router System")
    print("=" * 50)
    
    # Configuration
    config = {
        'db_params': {
            "host": "localhost",
            "database": "musteri_db",
            "user": "postgres",
            "password": "123",
            "port": "5432"
        },
        'pdf_path': "business_rules.pdf",
        'csv_path': "Data/goalscorers.csv"
    }
    
    try:
        # Setup LLM
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found")
        
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            api_key=groq_api_key,
            temperature=0.1
        )
        
        # Initialize router agent
        print("🔧 Initializing router agent...")
        router = RouterAgent(llm=llm, config=config)
        
        print("✅ Router agent ready!")
        print("\n📝 Example queries:")
        print("- 'What resources are available?'")
        print("- 'Analyze the sales data statistics'")
        print("- 'Find customers from database using business rules'")
        print("- 'What's the weather in Istanbul?'")
        print()
        
        # Main interaction loop
        while True:
            try:
                query = input("💬 Enter your query (or 'quit' to exit): ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    print("❌ Please enter a valid query")
                    continue
                
                # Process query through router
                print(f"\n🔍 Processing: {query}")
                result = router.process(query)
                
                # Display result
                if result.success:
                    print("✅ Router Result:")
                    if isinstance(result.data, dict) and 'output' in result.data:
                        print(result.data['output'])
                    else:
                        print(result.data)
                else:
                    print(f"❌ Error: {result.error}")
                
                print("\n" + "-" * 50 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
                
    except Exception as e:
        print(f"❌ System initialization failed: {e}")

if __name__ == "__main__":
    main()