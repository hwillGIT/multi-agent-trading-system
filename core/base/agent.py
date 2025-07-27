"""
Base agent class for all trading system agents.
"""

import abc
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from loguru import logger
from pydantic import BaseModel, Field

from .config import config
from .exceptions import TradingSystemError


class AgentOutput(BaseModel):
    """Standard output format for all agents."""
    
    agent_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None


class BaseAgent(abc.ABC):
    """
    Abstract base class for all trading system agents.
    
    All agents must implement the execute method and follow the standard
    input/output format for seamless integration.
    """
    
    def __init__(self, name: str, config_section: Optional[str] = None):
        """
        Initialize the agent.
        
        Args:
            name: Agent name for logging and identification
            config_section: Configuration section to load agent-specific settings
        """
        self.name = name
        self.config_section = config_section
        self.logger = logger.bind(agent=name)
        
        # Load agent-specific configuration
        if config_section:
            self.agent_config = config.get_section(config_section)
        else:
            self.agent_config = {}
        
        self.logger.info(f"Initialized agent: {name}")
    
    @abc.abstractmethod
    async def execute(self, inputs: Dict[str, Any]) -> AgentOutput:
        """
        Execute the agent's main logic.
        
        Args:
            inputs: Input data dictionary
            
        Returns:
            AgentOutput containing results and metadata
        """
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any], required_keys: List[str]) -> None:
        """
        Validate that required input keys are present.
        
        Args:
            inputs: Input dictionary to validate
            required_keys: List of required keys
            
        Raises:
            TradingSystemError: If required keys are missing
        """
        missing_keys = [key for key in required_keys if key not in inputs]
        if missing_keys:
            raise TradingSystemError(
                f"Missing required inputs for {self.name}: {missing_keys}"
            )
    
    def log_execution_start(self, inputs: Dict[str, Any]) -> datetime:
        """Log the start of agent execution."""
        start_time = datetime.utcnow()
        self.logger.info(
            f"Starting execution of {self.name}",
            extra={"inputs_keys": list(inputs.keys())}
        )
        return start_time
    
    def log_execution_end(self, start_time: datetime, success: bool = True) -> float:
        """
        Log the end of agent execution and return execution time.
        
        Args:
            start_time: When execution started
            success: Whether execution was successful
            
        Returns:
            Execution time in milliseconds
        """
        end_time = datetime.utcnow()
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        log_level = "info" if success else "error"
        getattr(self.logger, log_level)(
            f"Completed execution of {self.name}",
            extra={
                "execution_time_ms": execution_time_ms,
                "success": success
            }
        )
        
        return execution_time_ms
    
    async def safe_execute(self, inputs: Dict[str, Any]) -> AgentOutput:
        """
        Execute the agent with error handling and logging.
        
        Args:
            inputs: Input data dictionary
            
        Returns:
            AgentOutput with results or error information
        """
        start_time = self.log_execution_start(inputs)
        
        try:
            result = await self.execute(inputs)
            execution_time_ms = self.log_execution_end(start_time, success=True)
            result.execution_time_ms = execution_time_ms
            return result
            
        except Exception as e:
            execution_time_ms = self.log_execution_end(start_time, success=False)
            self.logger.error(f"Error in {self.name}: {str(e)}")
            
            return AgentOutput(
                agent_name=self.name,
                data={},
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time_ms
            )
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value for this agent.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if self.config_section:
            full_key = f"{self.config_section}.{key}"
            return config.get(full_key, default)
        return default