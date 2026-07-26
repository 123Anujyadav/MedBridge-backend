import logging.config
import sys
from app.core.config import settings

def setup_logging():
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if settings.ENVIRONMENT == "production":
        # Structured JSON formatter for production
        formatter_class = "pythonjsonlogger.jsonlogger.JsonFormatter"
        format_str = "%(timestamp)s %(level)s %(name)s %(message)s %(correlation_id)s"
    else:
        # Easy-to-read formatter for local development
        formatter_class = "logging.Formatter"
        format_str = log_format

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": format_str,
                "class": formatter_class,
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": sys.stdout,
            }
        },
        "root": {
            "level": settings.LOG_LEVEL,
            "handlers": ["console"],
        },
        "loggers": {
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.error": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "sqlalchemy.engine": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
            "celery": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            }
        }
    }

    logging.config.dictConfig(logging_config)
