import time
import logging
from typing import Any, Dict, Optional

from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from openai.types.responses.response_computer_tool_call import ResponseComputerToolCall

from ..types import CUAState, get_configuration_with_defaults
from ..utils import get_instance, is_computer_tool_call

# Set up logger
logger = logging.getLogger(__name__)

# Copied from the OpenAI example repository
# https://github.com/openai/openai-cua-sample-app/blob/eb2d58ba77ffd3206d3346d6357093647d29d99c/computers/scrapybara.py#L10
CUA_KEY_TO_SCRAPYBARA_KEY = {
    "/": "slash",
    "\\": "backslash",
    "arrowdown": "Down",
    "arrowleft": "Left",
    "arrowright": "Right",
    "arrowup": "Up",
    "backspace": "BackSpace",
    "capslock": "Caps_Lock",
    "cmd": "Meta_L",
    "delete": "Delete",
    "end": "End",
    "enter": "Return",
    "esc": "Escape",
    "home": "Home",
    "insert": "Insert",
    "option": "Alt_L",
    "pagedown": "Page_Down",
    "pageup": "Page_Up",
    "tab": "Tab",
    "win": "Meta_L",
}


def take_computer_action(state: CUAState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Executes computer actions based on the tool call in the last message using local Playwright.

    Args:
        state: The current state of the CUA agent.
        config: The runnable configuration.

    Returns:
        A dictionary with updated state information.
    """
    logger.info("Starting take_computer_action node execution", stack_info=True)
    message: AnyMessage = state.get("messages", [])[-1]
    assert message.type == "ai", "Last message must be an AI message"
    tool_outputs = message.additional_kwargs.get("tool_outputs")

    if not is_computer_tool_call(tool_outputs):
        # This should never happen, but include the check for proper type safety.
        raise ValueError("Cannot take computer action without a computer call in the last message.")

    # Cast tool_outputs as list[ResponseComputerToolCall] since is_computer_tool_call is true
    tool_outputs: list[ResponseComputerToolCall] = tool_outputs

    instance_id = state.get("instance_id")
    if not instance_id:
        raise ValueError("Instance ID not found in state.")
    
    # Get the local Playwright instance
    instance = get_instance(instance_id, config)
    if not instance:
        error_msg = f"Failed to get Playwright instance with ID: {instance_id}"
        logger.error(error_msg, stack_info=True)
        raise ValueError(error_msg)

    output = tool_outputs[-1]
    action = output.get("action")
    action_type = action.get("type")
    logger.info(f"Processing tool call: {action_type}", stack_info=True)
    tool_message: Optional[ToolMessage] = None

    try:
        computer_response = None
        # Action type already extracted above for logging

        # Define a helper function to wrap screenshot handling
        def handle_screenshot_response(response):
            import logging
            import base64
            logger = logging.getLogger(__name__)
            
            # Create static 1x1 PNG to use as fallback in case of failure
            FALLBACK_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVQI12P4//8/AAX+Av7czFnnAAAAAElFTkSuQmCC"
            
            try:
                # Extract screenshot from response
                if not response:
                    logger.error("Empty response received from computer action", stack_info=True)
                    screenshot = FALLBACK_PNG
                elif not hasattr(response, 'action_result'):
                    logger.error("Response missing action_result attribute", stack_info=True)
                    screenshot = FALLBACK_PNG
                else:
                    # Try to get screenshot from different possible locations
                    screenshot = response.action_result.get('screenshot')
                    if not screenshot and hasattr(response, 'base_64_image'):
                        screenshot = response.base_64_image
                        
                    if not screenshot:
                        logger.error(f"No screenshot found in response: {response.action_result}", stack_info=True)
                        screenshot = FALLBACK_PNG
                
                # For OpenAI's computer-use-preview, the response MUST have this format
                # with image_url containing a valid data URL
                return {
                    "role": "tool",
                    "content": [{
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{screenshot}",
                    }],
                    "tool_call_id": output.get("call_id"),
                    "additional_kwargs": {"type": "computer_call_output"},
                }
            except Exception as e:
                logger.error(f"Error processing screenshot: {str(e)}", stack_info=True)
                # Always return a valid response even if processing fails
                return {
                    "role": "tool",
                    "content": [{
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{FALLBACK_PNG}",
                    }],
                    "tool_call_id": output.get("call_id"),
                    "additional_kwargs": {"type": "computer_call_output"},
                }

        if action_type == "click":
            # Execute click action in Playwright
            button = "middle" if action.get("button") == "wheel" else action.get("button", "left")
            coords = [action.get("x", 0), action.get("y", 0)]
            logger.info(f"TOOL CALL: click_mouse at coordinates {coords} with button {button}", stack_info=True)
            
            computer_response = instance.computer(
                action="click_mouse",
                button=button,
                coordinates=coords
            )
            tool_message = handle_screenshot_response(computer_response)
        
        elif action_type == "keypress":
            # Map keys to Playwright format
            keys = action.get("keys", [])
            logger.info(f"TOOL CALL: press_key with keys {keys}", stack_info=True)
            
            # Use the computer method to handle keypress
            computer_response = instance.computer(
                action="press_key",
                keys=keys
            )
            tool_message = handle_screenshot_response(computer_response)
            
        elif action_type == "screenshot":
            logger.info("TOOL CALL: take_screenshot", stack_info=True)
            computer_response = instance.computer(action="take_screenshot")
            tool_message = handle_screenshot_response(computer_response)
        
        elif action_type == "wait":
            # Pass the wait action to the computer method which handles it non-blockingly
            logger.info("TOOL CALL: wait", stack_info=True)
            computer_response = instance.computer(action="wait")
            tool_message = handle_screenshot_response(computer_response)
        
        elif action_type == "type":
            # Type text
            text = action.get("text", "")
            logger.info(f"TOOL CALL: type_text with content '{text}'", stack_info=True)
            
            computer_response = instance.computer(
                action="type_text",
                text=text
            )
            tool_message = handle_screenshot_response(computer_response)
        
        elif action_type == "scroll":
            # Scroll the page
            delta_x = action.get("scroll_x", 0) // 20
            delta_y = action.get("scroll_y", 0) // 20
            coords = [action.get("x", 0), action.get("y", 0)]
            logger.info(f"TOOL CALL: scroll with delta_x={delta_x}, delta_y={delta_y} at coordinates {coords}", stack_info=True)
            
            computer_response = instance.computer(
                action="scroll",
                delta_x=delta_x,
                delta_y=delta_y,
                coordinates=coords
            )
            tool_message = handle_screenshot_response(computer_response)
            
        else:
            error_msg = f"Action type not yet implemented for local Playwright: {action_type}"
            logger.error(error_msg, stack_info=True)
            
            # Take a screenshot anyway to avoid graph errors
            logger.info("TOOL CALL: taking fallback screenshot after unsupported action", stack_info=True)
            computer_response = instance.computer(action="take_screenshot")
            tool_message = handle_screenshot_response(computer_response)

    except Exception as e:
        error_msg = f"Failed to execute computer call: {e}"
        logger.error(error_msg, stack_info=True)
        logger.error(f"Computer call details: {output}", stack_info=True)
        
        try:
            # Try to take a screenshot to recover from error
            logger.info("TOOL CALL: taking recovery screenshot after error", stack_info=True)
            computer_response = instance.computer(action="take_screenshot")
            tool_message = handle_screenshot_response(computer_response)
        except Exception as screenshot_error:
            logger.error(f"Failed to take recovery screenshot: {screenshot_error}", stack_info=True)

    # Always ensure we return a valid messages value to avoid 'left/right' error
    # If we don't have a tool_message, create a fallback one with error information
    if not tool_message:
        tool_message = {
            "role": "tool",
            "content": [{"type": "text", "text": f"Action '{action_type}' could not be completed"}],
            "tool_call_id": output.get("call_id"),
            "additional_kwargs": {"type": "computer_call_output"},
        }

    logger.info(f"TOOL CALL COMPLETED: {action_type}", stack_info=True)
    return {
        "messages": tool_message,  # Never return None for messages
        "instance_id": instance.id if instance else None,
    }
