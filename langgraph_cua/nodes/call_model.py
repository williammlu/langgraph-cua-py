from typing import Any, Dict, Optional, Union

from langchain_core.messages import AIMessageChunk, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI

from ..types import CUAState, get_configuration_with_defaults


def get_openai_env_from_state_env(env: str) -> str:
    """
    Converts one of "web", "ubuntu", or "windows" to OpenAI environment string.

    Args:
        env: The environment to convert.

    Returns:
        The corresponding OpenAI environment string.

    Raises:
        ValueError: If the environment is invalid.
    """
    if env == "web":
        return "browser"
    elif env == "ubuntu":
        return "ubuntu"
    elif env == "windows":
        return "windows"


DEFAULT_DISPLAY_WIDTH = 1024
DEFAULT_DISPLAY_HEIGHT = 768


def _prompt_to_sys_message(prompt: Union[str, SystemMessage, None]):
    if prompt is None:
        return None
    if isinstance(prompt, str):
        return {"role": "system", "content": prompt}
    return prompt


async def call_model(state: CUAState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Invokes the computer preview model with the given messages.

    Args:
        state: The current state of the thread.

    Returns:
        The updated state with the model's response.
    """
    import logging
    logger = logging.getLogger(__name__)
    configuration = get_configuration_with_defaults(config)
    environment = configuration.get("environment")
    zdr_enabled = configuration.get("zdr_enabled")
    prompt = _prompt_to_sys_message(configuration.get("prompt"))
    messages = state.get("messages", [])
    previous_response_id: Optional[str] = None
    last_message = messages[-1] if messages else None

    # Check if the last message is a tool message
    if last_message and getattr(last_message, "type", None) == "tool" and zdr_enabled is False:
        # If it's a tool message, check if the second-to-last message is an AI message
        if (
            len(messages) >= 2
            and getattr(messages[-2], "type", None) == "ai"
            and hasattr(messages[-2], "response_metadata")
        ):
            previous_response_id = messages[-2].response_metadata["id"]

    llm = ChatOpenAI(
        model="computer-use-preview",
        model_kwargs={"truncation": "auto", "previous_response_id": previous_response_id},
    )

    tool = {
        "type": "computer_use_preview",
        "display_width": DEFAULT_DISPLAY_WIDTH,
        "display_height": DEFAULT_DISPLAY_HEIGHT,
        "environment": get_openai_env_from_state_env(environment),
    }
    llm_with_tools = llm.bind_tools([tool])

    response = None
    
    try:
        # Check if the last message is a tool message
        if last_message and getattr(last_message, "type", None) == "tool" and zdr_enabled is False:
            if previous_response_id is None:
                logger.error("Cannot process tool message without a previous_response_id", stack_info=True)
                # Create a fallback response instead of raising an exception
                from langchain_core.messages import AIMessage
                response = AIMessage(content="I'm having trouble processing your request with the browser. Please try again or provide a different instruction.")
            else:
                # Only pass the tool message to the model
                response = await llm_with_tools.ainvoke([last_message])
        else:
            # Pass all messages to the model
            if prompt is None:
                response = await llm_with_tools.ainvoke(messages)
            else:
                response = await llm_with_tools.ainvoke([prompt, *messages])
    except Exception as e:
        logger.error(f"Error calling OpenAI model: {str(e)}", stack_info=True)
        # Create a fallback response
        from langchain_core.messages import AIMessage
        response = AIMessage(content="I encountered an error when trying to access the browser. Please check your API keys and try again.")
    
    if response is None:
        logger.error("No response was generated", stack_info=True)
        from langchain_core.messages import AIMessage
        response = AIMessage(content="No response was generated. Please try again.")

    return {
        "messages": response,
    }
