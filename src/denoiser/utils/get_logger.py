import logging
from pathlib import Path
from typing import Literal


def create_logger(name: str, log_dir: Path, verbose_level: Literal["debug", "info", "error"]) -> logging.Logger:
    """Create and return a logger with the specified name and verbosity level.

    Args:
        name: Logger name
        log_dir: Directory path where log files will be saved
        verbose_level: Logging level ("debug", "info", "error")

    Returns:
        Configured logger that outputs to both console and file
    """
    level_mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "error": logging.ERROR,
    }

    # Get the logger
    logger = logging.getLogger(name)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Set the logging level
    log_level = level_mapping.get(verbose_level, logging.INFO)
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create log directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create file handler
    log_file_path = log_dir / f"{name}.log"
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


if __name__ == "__main__":
    # Example usage
    logger = create_logger("my_logger", Path("./logs"), "debug")
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.error("This is an error message.")
