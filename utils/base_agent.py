from typing import Optional, Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from collections.abc import Iterable


# LangChain imports
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
# Pydantic imports
from pydantic import BaseModel, Field, validator
from utils.loggerCenter import LoggerCenter

logger = LoggerCenter().get_logger()

load_dotenv()

class CodeOutput(BaseModel):
    """Pydantic model for structured code output"""
    code: str = Field(description="Python pandas/SQL code that assigns the result to 'result' variable.")


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
    #     self._setup_tools()
    #     self._setup_agent()

    # @abstractmethod
    # def _get_system_prompt(self) -> str:
    #     """Get the system prompt for this specialized agent"""
    #     pass
    
    # @abstractmethod
    # def _setup_tools(self):
    #     """Setup specialized tools for this agent"""
    #     pass
        
    # def _setup_prompt(self):
    #     """Setup the prompt template for the agent"""
    #     prompt = ChatPromptTemplate.from_messages([
    #         ("system", self._get_system_prompt()),
    #         ("user", "{input}"),
    #         ("placeholder", "{agent_scratchpad}")
    #     ])
    #     return prompt
    
    # def _setup_agent(self):
    #     """Setup the agent with specialized tools"""
    #     try:
    #         agent = create_tool_calling_agent(self.llm, self.tools, self._setup_prompt())
    #         self.agent_executor = AgentExecutor(
    #             agent=agent, 
    #             tools=self.tools, 
    #             verbose=True,
    #             handle_parsing_errors=True,
    #             max_iterations=4, 
    #         )
    #         logger.info(f"Agent {self.agent_name} setup completed successfully")
    #     except Exception as e:
    #         logger.error(f"Agent {self.agent_name} setup failed: {e}")
    #         raise
    
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


def to_dataframe_safe(data, column_name="value"):
    if isinstance(data, dict):
        return pd.DataFrame(data)
    if isinstance(data, pd.Series):
        return data.to_frame(name=data.name if data.name else column_name)
    if isinstance(data, (pd.Index, np.ndarray)):
        data = data.tolist()
    if not isinstance(data, Iterable) or isinstance(data, (str, bytes)):
        data = [data]
    if isinstance(data, (set, range)) or hasattr(data, "__next__"):
        data = list(data)
    if isinstance(data, list) and all(isinstance(i, (list, tuple)) for i in data):
        return pd.DataFrame(data)
    return pd.DataFrame(data, columns=[column_name])

def is_safe_code( code: str) -> bool:
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

def clean_imports_from_code(code: str) -> str:
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
