import os
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict, Union

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import add_messages


class Output(TypedDict):
    """
    A computer screenshot image used with the computer use tool.
    """

    type: Literal["computer_screenshot"]  # Always "computer_screenshot"
    file_id: Optional[str]  # The identifier of an uploaded file that contains the screenshot
    image_url: Optional[str]  # The URL of the screenshot image


class AcknowledgedSafetyCheck(TypedDict):
    """
    A pending safety check for the computer call.
    """

    id: str  # The ID of the pending safety check
    code: str  # The type of the pending safety check
    message: str  # Details about the pending safety check


class ComputerCallOutput(TypedDict):
    """
    The output of a computer tool call.
    """

    call_id: str  # The ID of the computer tool call that produced the output
    output: Output  # A computer screenshot image used with the computer use tool
    type: Literal["computer_call_output"]  # Always "computer_call_output"
    id: Optional[str]  # The ID of the computer tool call output
    acknowledged_safety_checks: Optional[
        List[AcknowledgedSafetyCheck]
    ]  # Safety checks acknowledged by the developer
    status: Optional[
        Literal["in_progress", "completed", "incomplete"]
    ]  # Status of the message input


class CUAState(TypedDict):
    """State schema for the computer use agent.

    Attributes:
        messages: The messages between the user and assistant.
        instance_id: The ID of the instance to use for this thread.

        stream_url: The URL to the live-stream of the virtual machine.
        authenticated_id: The ID of the auth state currently in use.
    """

    messages: Annotated[list[AnyMessage], add_messages] = []
    instance_id: Annotated[Optional[str], None] = None
    stream_url: Annotated[Optional[str], None] = None
    authenticated_id: Annotated[Optional[str], None] = None


class CUAConfiguration(TypedDict):
    """Configuration for the Computer Use Agent.

    Attributes:
        timeout_hours: The number of hours to keep the virtual machine running before it times out.
            Must be between 0.01 and 24. Default is 1.
        zdr_enabled: Whether or not Zero Data Retention is enabled in the user's OpenAI account. If True,
            the agent will not pass the 'previous_response_id' to the model, and will always pass it the full
            message history for each request. If False, the agent will pass the 'previous_response_id' to the
            model, and only the latest message in the history will be passed. Default False.
        auth_state_id: The ID of the authentication state. If defined, it will be used to authenticate
            with Scrapybara. Only applies if 'environment' is set to 'web'.
        environment: The environment to use. Default is "web".
        prompt: The initial prompt to use for the conversation. Will
            be passed as a system message.
        use_local_playwright: Whether to use a local Playwright instance instead of Scrapybara.
            If True, a local Playwright browser will be used. Default is False.
        headless: Whether to run the local Playwright browser in headless mode.
            Only used if use_local_playwright is True. Default is False.
        browser_type: The type of browser to use with local Playwright.
            Options are "chromium", "firefox", or "webkit". Default is "chromium".
            Only used if use_local_playwright is True.
    """

    timeout_hours: Optional[float]  # Timeout in hours (0.01-24, default: 1)
    zdr_enabled: Optional[bool]  # True/False for whether or not ZDR is enabled.
    auth_state_id: Optional[str]  # The ID of the authentication state.
    environment: Optional[
        Literal["web", "ubuntu", "windows"]
    ]  # The environment to use. Default is "web".
    prompt: Optional[Union[str, SystemMessage]]  # The initial prompt to use for the conversation
    use_local_playwright: Optional[bool]  # Whether to use a local Playwright instance
    headless: Optional[bool]  # Whether to run the browser in headless mode
    browser_type: Optional[Literal["chromium", "firefox", "webkit"]]  # Browser type for local Playwright


def get_configuration_with_defaults(config: RunnableConfig) -> Dict[str, Any]:
    """
    Gets the configuration with defaults for the graph.

    Args:
        config: The configuration for the runnable.

    Returns:
        Dict with configuration values including defaults.
    """

    configurable_fields = config.get("configurable", {})
    timeout_hours = configurable_fields.get("timeout_hours", 1)
    zdr_enabled = configurable_fields.get("zdr_enabled", False)
    auth_state_id = configurable_fields.get("auth_state_id", None)
    environment = configurable_fields.get("environment", "web")
    prompt = configurable_fields.get("prompt", None)
    use_local_playwright = configurable_fields.get("use_local_playwright", False)
    headless = configurable_fields.get("headless", False)
    browser_type = configurable_fields.get("browser_type", "chromium")

    return {
        "timeout_hours": timeout_hours,
        "zdr_enabled": zdr_enabled,
        "auth_state_id": auth_state_id,
        "environment": environment,
        "prompt": prompt,
        "use_local_playwright": use_local_playwright,
        "headless": headless,
        "browser_type": browser_type,
    }
