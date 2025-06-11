
import logging
from functools import wraps
from typing import Callable, Any

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServiceError(Exception):
    """Base exception for service errors."""
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class LLMServiceError(ServiceError):
    """LLM service specific errors."""
    pass

class RAGServiceError(ServiceError):
    """RAG service specific errors."""
    pass

class DatabaseError(ServiceError):
    """Database operation errors."""
    pass

def handle_service_errors(func: Callable) -> Callable:
    """Decorator to handle service errors consistently."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except ServiceError:
            # Re-raise service errors as-is
            raise
        except ValueError as e:
            logger.error(f"Validation error in {func.__name__}: {e}")
            raise ServiceError(f"Invalid input: {e}", "VALIDATION_ERROR")
        except ConnectionError as e:
            logger.error(f"Connection error in {func.__name__}: {e}")
            raise ServiceError(f"Service unavailable: {e}", "CONNECTION_ERROR")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise ServiceError(f"Internal service error: {e}", "INTERNAL_ERROR")
    return wrapper