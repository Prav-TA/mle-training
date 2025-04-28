import logging
import os


def get_project_root():
    """
    Return the root directory of the project.

    Returns
    -------
    str
        The project root directory.
    """

    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging(log_level="INFO", log_path=None, no_console=False):
    """
    Configure logging for the application.

    This function sets up logging with a specified log level and optional file and console outputs.

    Parameters
    ----------
    log_level : str, optional
        The level of logging (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    log_path : str or None, optional
        The file path to write log messages to. If None, logging to a file is not performed.
    no_console : bool, optional
        If True, suppress console logging. Default is False.

    Returns
    -------
    None
    """

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = []
    if not no_console:
        handlers.append(logging.StreamHandler())
    if log_path:
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
