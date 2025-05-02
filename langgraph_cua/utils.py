from typing import Any, Union, Optional
import logging

from langchain_core.runnables import RunnableConfig

# Import only from our local Playwright client
from .playwright_client import PlaywrightClient, BrowserInstance, UbuntuInstance, WindowsInstance
from .types import get_configuration_with_defaults

# Set up logger
logger = logging.getLogger(__name__)

# Singleton instance of PlaywrightClient to ensure we reuse the same client
_playwright_client_instance = None


def get_browser_client(use_local_playwright: bool = True):
    """
    Gets a local Playwright client.
    Uses a singleton pattern to ensure we reuse the same client instance.

    Args:
        use_local_playwright: Should always be True now that we only support local Playwright.

    Returns:
        A PlaywrightClient instance.
    """
    global _playwright_client_instance
    
    if _playwright_client_instance is None:
        logger.info("Creating new PlaywrightClient instance")
        print("Using local Playwright")
        _playwright_client_instance = PlaywrightClient()
    
    return _playwright_client_instance


def get_instance(id: str, config: RunnableConfig) -> Union[BrowserInstance, UbuntuInstance, WindowsInstance]:
    """
    Gets an instance by its ID from local Playwright.

    Args:
        id: The ID of the instance to get.
        config: The configuration for the runnable.

    Returns:
        The instance.
    """
    client = get_browser_client()
    return client.get(id)


def is_computer_tool_call(tool_outputs: Any) -> bool:
    """
    Checks if the given tool outputs are a computer call.

    Args:
        tool_outputs: The tool outputs to check.

    Returns:
        True if the tool outputs are a computer call, false otherwise.
    """
    if not tool_outputs or not isinstance(tool_outputs, list):
        return False

    return any(output.get("type") == "computer_call" for output in tool_outputs)
