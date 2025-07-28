from typing import Optional, Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime
from abc import ABC, abstractmethod

# LangChain imports
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
# Pydantic imports
from pydantic import BaseModel, Field, validator
from loggerCenter import LoggerCenter

logger = LoggerCenter().get_logger()

# Load environment variables
load_dotenv()

class CodeOutput(BaseModel):
    """Pydantic model for structured code output"""
    code: str = Field(description="Python pandas/SQL code that assigns the result to 'result' variable.")
    explanation: Optional[str] = Field(description="Brief explanation of what the code does", default=None)
    data_source: Optional[str] = Field(description="Source of data (csv, postgres, api)", default="csv")

class AgentResult:
    """Container for agent results"""
    def __init__(self, success: bool, data: Any = None, error: str = None, metadata: Dict = None):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}

class BaseSpecializedAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, agent_name: str, llm: ChatGroq):
        self.agent_name = agent_name
        self.llm = llm
        self.tools = []
        self.agent_executor = None
        self._setup_tools()
        self._setup_agent()

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this specialized agent"""
        pass
    
    @abstractmethod
    def _setup_tools(self):
        """Setup specialized tools for this agent"""
        pass
        
    def _setup_prompt(self):
        """Setup the prompt template for the agent"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),  # Bu önemli - "assistant" yerine "placeholder" olmalı
        ])
        return prompt
    
    def _setup_agent(self):
        """Setup the agent with its specialized tools"""
        try:
            agent = create_tool_calling_agent(self.llm, self.tools, self._setup_prompt())
            self.agent_executor = AgentExecutor(
                agent=agent, 
                tools=self.tools, 
                verbose=True,
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
            logger.info(f"Agent {self.agent_name} setup completed successfully")
        except Exception as e:
            logger.error(f"Agent {self.agent_name} setup failed: {e}")
            raise
    
    @abstractmethod
    def process(self, *args, **kwargs) -> AgentResult:
        """Process the input and return AgentResult"""
        pass
        
    def log_info(self, message: str):
        logger.info(f"[{self.agent_name}] {message}")
    
    def log_error(self, message: str):
        logger.error(f"[{self.agent_name}] {message}")
    
    def log_warning(self, message: str):
        logger.warning(f"[{self.agent_name}] {message}")

