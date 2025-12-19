"""
Common configuration and environment setup for Ops Copilot.

This module centralizes all configuration including:
- API keys and credentials
- Model settings
- Database connections
- Service endpoints
"""

import os
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Uses pydantic BaseSettings to automatically load from .env file
    and validate configuration values.
    """
    
    # ============================================================================
    # LLM API Keys
    # ============================================================================
    
    openai_api_key: str = Field(
        ...,  # Required field
        env="OPENAI_API_KEY",
        description="OpenAI API key for GPT-4 access"
    )
    
    anthropic_api_key: Optional[str] = Field(
        None,
        env="ANTHROPIC_API_KEY",
        description="Anthropic API key for Claude access (optional)"
    )
    
    # ============================================================================
    # LLM Model Configuration
    # ============================================================================
    
    llm_model: str = Field(
        default="gpt-4-turbo-preview",
        env="LLM_MODEL",
        description="Primary LLM model to use"
    )
    
    llm_temperature: float = Field(
        default=0.1,
        env="LLM_TEMPERATURE",
        description="Temperature for LLM responses (0.0 = deterministic, 1.0 = creative)"
    )
    
    llm_max_tokens: int = Field(
        default=2000,
        env="LLM_MAX_TOKENS",
        description="Maximum tokens for LLM responses"
    )
    
    # ============================================================================
    # LangSmith (Observability)
    # ============================================================================
    
    langchain_tracing_v2: bool = Field(
        default=True,
        env="LANGCHAIN_TRACING_V2",
        description="Enable LangSmith tracing"
    )
    
    langchain_project: str = Field(
        default="ops-copilot",
        env="LANGCHAIN_PROJECT",
        description="LangSmith project name"
    )
    
    langchain_api_key: Optional[str] = Field(
        None,
        env="LANGCHAIN_API_KEY",
        description="LangSmith API key"
    )
    
    # ============================================================================
    # Vector Database (Pinecone)
    # ============================================================================
    
    pinecone_api_key: str = Field(
        ...,
        env="PINECONE_API_KEY",
        description="Pinecone API key for vector storage"
    )
    
    pinecone_environment: str = Field(
        default="us-west1-gcp",
        env="PINECONE_ENVIRONMENT",
        description="Pinecone environment/region"
    )
    
    pinecone_index_name: str = Field(
        default="runbooks",
        env="PINECONE_INDEX_NAME",
        description="Pinecone index name for runbook embeddings"
    )
    
    # ============================================================================
    # Elasticsearch (Logs & Keyword Search)
    # ============================================================================
    
    elasticsearch_url: str = Field(
        default="http://localhost:9200",
        env="ELASTICSEARCH_URL",
        description="Elasticsearch cluster URL"
    )
    
    elasticsearch_username: Optional[str] = Field(
        None,
        env="ELASTICSEARCH_USERNAME",
        description="Elasticsearch username (if auth enabled)"
    )
    
    elasticsearch_password: Optional[str] = Field(
        None,
        env="ELASTICSEARCH_PASSWORD",
        description="Elasticsearch password (if auth enabled)"
    )
    
    # ============================================================================
    # PostgreSQL (Incident Storage)
    # ============================================================================
    
    postgres_host: str = Field(
        default="localhost",
        env="POSTGRES_HOST",
        description="PostgreSQL host"
    )
    
    postgres_port: int = Field(
        default=5432,
        env="POSTGRES_PORT",
        description="PostgreSQL port"
    )
    
    postgres_database: str = Field(
        default="opscopilot",
        env="POSTGRES_DATABASE",
        description="PostgreSQL database name"
    )
    
    postgres_username: str = Field(
        default="postgres",
        env="POSTGRES_USERNAME",
        description="PostgreSQL username"
    )
    
    postgres_password: str = Field(
        ...,
        env="POSTGRES_PASSWORD",
        description="PostgreSQL password"
    )
    
    # ============================================================================
    # Redis (Caching)
    # ============================================================================
    
    redis_host: str = Field(
        default="localhost",
        env="REDIS_HOST",
        description="Redis host"
    )
    
    redis_port: int = Field(
        default=6379,
        env="REDIS_PORT",
        description="Redis port"
    )
    
    redis_password: Optional[str] = Field(
        None,
        env="REDIS_PASSWORD",
        description="Redis password (if auth enabled)"
    )
    
    redis_ttl: int = Field(
        default=3600,
        env="REDIS_TTL",
        description="Default TTL for cached items (seconds)"
    )
    
    # ============================================================================
    # Prometheus (Metrics)
    # ============================================================================
    
    prometheus_url: str = Field(
        default="http://localhost:9090",
        env="PROMETHEUS_URL",
        description="Prometheus server URL"
    )
    
    # ============================================================================
    # Application Settings
    # ============================================================================
    
    max_iterations: int = Field(
        default=5,
        env="MAX_ITERATIONS",
        description="Maximum agent iterations to prevent infinite loops"
    )
    
    request_timeout: int = Field(
        default=30,
        env="REQUEST_TIMEOUT",
        description="Maximum request processing time (seconds)"
    )
    
    embedding_model: str = Field(
        default="text-embedding-3-large",
        env="EMBEDDING_MODEL",
        description="OpenAI embedding model"
    )
    
    embedding_dimensions: int = Field(
        default=1536,
        env="EMBEDDING_DIMENSIONS",
        description="Embedding vector dimensions"
    )
    
    # ============================================================================
    # Configuration
    # ============================================================================
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    # ============================================================================
    # Helper Methods
    # ============================================================================
    
    @property
    def postgres_url(self) -> str:
        """
        Construct PostgreSQL connection URL.
        
        Returns:
            str: PostgreSQL connection string
        """
        return (
            f"postgresql://{self.postgres_username}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
        )
    
    @property
    def redis_url(self) -> str:
        """
        Construct Redis connection URL.
        
        Returns:
            str: Redis connection string
        """
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"
        return f"redis://{self.redis_host}:{self.redis_port}/0"


# ============================================================================
# Global Settings Instance
# ============================================================================

# Create a singleton settings instance
# This will be imported by other modules
settings = Settings()


# ============================================================================
# Environment Setup Functions
# ============================================================================

def setup_environment() -> None:
    """
    Setup environment variables for LangChain and other services.
    
    This function should be called at application startup to ensure
    all necessary environment variables are set.
    """
    # Set OpenAI API key
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    
    # Set LangSmith tracing
    if settings.langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        if settings.langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    
    # Set Anthropic API key if available
    if settings.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key


def validate_configuration() -> bool:
    """
    Validate that all required configuration is present.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    
    Raises:
        ValueError: If critical configuration is missing
    """
    errors = []
    
    # Check required API keys
    if not settings.openai_api_key:
        errors.append("OPENAI_API_KEY is required")
    
    if not settings.pinecone_api_key:
        errors.append("PINECONE_API_KEY is required")
    
    if not settings.postgres_password:
        errors.append("POSTGRES_PASSWORD is required")
    
    # Raise error if any required fields are missing
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    return True


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Example: Load and validate configuration
    try:
        validate_configuration()
        setup_environment()
        print("✅ Configuration loaded successfully")
        print(f"   LLM Model: {settings.llm_model}")
        print(f"   Max Iterations: {settings.max_iterations}")
        print(f"   Request Timeout: {settings.request_timeout}s")
        print(f"   Embedding Model: {settings.embedding_model}")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
