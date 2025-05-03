"""
Local Playwright implementation to replace Scrapybara.
This module provides a drop-in replacement for Scrapybara that runs a local Playwright instance.
"""

import asyncio
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from playwright.sync_api import sync_playwright, Browser, Page


@dataclass
class InstanceGetStreamUrlResponse:
    """A response from getting a stream URL for an instance."""
    stream_url: str


@dataclass
class ComputerResponse:
    """A response from a computer action."""
    action_result: Dict[str, Any]
    base_64_image: Optional[str] = None


class PlaywrightBrowserInstance:
    """
    A browser instance running in Playwright locally.
    This class mimics the Scrapybara BrowserInstance interface.
    """

    def __init__(self, instance_id: str, browser: Browser, page: Page):
        """
        Initialize a PlaywrightBrowserInstance.
        
        Args:
            instance_id: A unique ID for this instance
            browser: The Playwright browser instance
            page: The main page in the browser
        """
        self.id = instance_id
        self._browser = browser
        self._page = page

    def close(self):
        """Close the browser instance."""
        try:
            self._browser.close()
        except Exception as e:
            print(f"Error closing browser: {e}", stack_info=True)


    def get_page(self) -> Page:
        """
        Get the main page in the browser.
        
        Returns:
            The Playwright page instance
        """
        return self._page

    def computer_navigate(self, url: str) -> ComputerResponse:
        """
        Navigate to the given URL.
        
        Args:
            url: The URL to navigate to
            
        Returns:
            A ComputerResponse with the action result
        """
        try:
            self._page.goto(url)
            return ComputerResponse(action_result={"success": True, "current_url": self._page.url})
        except Exception as e:
            return ComputerResponse(action_result={"success": False, "error": str(e)})

    def computer_click(self, selector: str) -> ComputerResponse:
        """
        Click on the given selector.
        
        Args:
            selector: The selector to click on
            
        Returns:
            A ComputerResponse with the action result
        """
        try:
            self._page.click(selector)
            return ComputerResponse(action_result={"success": True})
        except Exception as e:
            return ComputerResponse(action_result={"success": False, "error": str(e)})

    def computer_get_elements(self, selector: str) -> ComputerResponse:
        """
        Get elements matching the given selector.
        
        Args:
            selector: The selector to find elements for
            
        Returns:
            A ComputerResponse with the action result
        """
        try:
            elements = self._page.query_selector_all(selector)
            element_data = []
            for el in elements:
                try:
                    text = el.text_content() or ""
                    tag_name = el.evaluate("el => el.tagName.toLowerCase()")
                    element_data.append({
                        "tag_name": tag_name,
                        "text": text,
                        "attributes": el.evaluate("el => Object.assign({}, ...Array.from(el.attributes).map(attr => ({[attr.name]: attr.value})))"),
                    })
                except Exception:
                    # Skip elements that can't be processed
                    pass
            
            return ComputerResponse(action_result={"success": True, "elements": element_data})
        except Exception as e:
            return ComputerResponse(action_result={"success": False, "error": str(e)})

    def computer_type(self, selector: str, text: str) -> ComputerResponse:
        """
        Type text into the element matching the given selector.
        
        Args:
            selector: The selector to type into
            text: The text to type
            
        Returns:
            A ComputerResponse with the action result
        """
        try:
            self._page.fill(selector, text)
            return ComputerResponse(action_result={"success": True})
        except Exception as e:
            return ComputerResponse(action_result={"success": False, "error": str(e)})

    def computer_screenshot(self) -> ComputerResponse:
        """
        Take a screenshot of the page.
        
        Returns:
            A ComputerResponse with the action result containing base64 encoded image
        """
        import logging
        import base64
        import os
        logger = logging.getLogger(__name__)
        
        logger.info("Taking screenshot of current page", stack_info=True)
        
        # Use a non-blocking approach instead of sleeping
        try:
            # Take the screenshot with specific options for better compatibility
            # OpenAI recommends images between 512x512 and 2048x2048 pixels
            # PNG format offers higher quality than JPEG for text and UI elements
            screenshot = self._page.screenshot(type="png", full_page=True)
            
            # Convert to base64 for transmission
            base64_image = base64.b64encode(screenshot).decode("utf-8")
            
            # Validate that we have a proper base64 image
            if base64_image and len(base64_image) > 100:  # Ensure it's not an empty or tiny image
                logger.info(f"Successfully took screenshot, size: {len(base64_image)} bytes")
                return ComputerResponse(
                    action_result={"success": True, "screenshot": base64_image}, 
                    base_64_image=base64_image
                )
            else:
                logger.error(f"Screenshot too small or empty: {len(base64_image) if base64_image else 0} bytes", stack_info=True)
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}", stack_info=True)
        
        # If screenshot failed, create a small fallback image
        try:
            # Create a valid 1x1 transparent PNG as fallback
            # This is a properly encoded 1x1 transparent PNG
            fallback_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
            logger.warning("Using fallback 1x1 PNG image since screenshot failed", stack_info=True)
            return ComputerResponse(
                action_result={"success": True, "screenshot": fallback_image}, 
                base_64_image=fallback_image
            )
        except Exception as e:
            error_msg = f"Error performing computer action take_screenshot: {e}"
            logger.error(error_msg, stack_info=True)
            logger.error(f"Action parameters: None", stack_info=True)
            import traceback
            logger.error(f"Exception traceback: {traceback.format_exc()}", stack_info=True)
            response = ComputerResponse(action_result={"success": False, "error": error_msg})
            return response
            
    def computer(self, action: str, **kwargs) -> ComputerResponse:
        """
        General purpose method to handle various computer actions.
        This method is called by the CUA agent to handle computer actions.
        
        Args:
            action: The action to perform (e.g., 'take_screenshot', 'click_mouse', etc.)
            **kwargs: Additional arguments for the action
            
        Returns:
            A ComputerResponse with the action result
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"BROWSER ACTION: {action} with parameters: {kwargs}", stack_info=True)
        
        try:
            if action == "take_screenshot":
                return self.computer_screenshot()
                
            elif action == "click_mouse":
                # Extract parameters
                coordinates = kwargs.get("coordinates", [0, 0])
                button = kwargs.get("button", "left")
                num_clicks = kwargs.get("num_clicks", 1)
                
                logger.info(f"Clicking at coordinates {coordinates} with button {button}, clicks: {num_clicks}", stack_info=True)
                
                # Click at the specified coordinates
                self._page.mouse.click(coordinates[0], coordinates[1], button=button, click_count=num_clicks)
                
                # Take a screenshot to return after the action
                return self.computer_screenshot()
                
            elif action == "type_text":
                # Type the text
                text = kwargs.get("text", "")
                logger.info(f"Typing text: '{text}'", stack_info=True)
                self._page.keyboard.type(text)
                
                # Take a screenshot to return after the action
                return self.computer_screenshot()
                
            elif action == "press_key":
                # Press the specified keys
                keys = kwargs.get("keys", [])
                logger.info(f"Pressing keys: {keys}", stack_info=True)
                for key in keys:
                    logger.info(f"Pressing key: {key}", stack_info=True)
                    self._page.keyboard.press(key)
                
                # Take a screenshot to return after the action
                screenshot = self.computer_screenshot()
                return screenshot
                
            elif action == "move_mouse":
                # Move the mouse to the specified coordinates
                coordinates = kwargs.get("coordinates", [0, 0])
                logger.info(f"Moving mouse to coordinates: {coordinates}", stack_info=True)
                self._page.mouse.move(coordinates[0], coordinates[1])
                
                # Take a screenshot to return after the action
                screenshot = self.computer_screenshot()
                return screenshot
                
            elif action == "drag_mouse":
                # Perform a drag operation
                start = kwargs.get("start", [0, 0])
                end = kwargs.get("end", [100, 100])
                button = kwargs.get("button", "left")
                
                logger.info(f"Dragging from {start} to {end} with button {button}", stack_info=True)
                
                # Drag from start to end
                self._page.mouse.move(start[0], start[1])
                self._page.mouse.down(button=button)
                self._page.mouse.move(end[0], end[1])
                self._page.mouse.up(button=button)
                
                # Take a screenshot to return after the action
                return self.computer_screenshot()
                
            elif action == "scroll":
                # Execute the scroll command
                delta_x = kwargs.get("delta_x", 0)
                delta_y = kwargs.get("delta_y", 0)
                coordinates = kwargs.get("coordinates", [0, 0])
                
                logger.info(f"Scrolling with delta_x={delta_x}, delta_y={delta_y} at coordinates {coordinates}", stack_info=True)
                
                # First move the mouse to the coordinates if specified
                if coordinates and coordinates != [0, 0]:
                    self._page.mouse.move(coordinates[0], coordinates[1])
                    
                # Then scroll
                for i in range(20):  # Break into smaller scrolls
                    logger.info(f"Scroll iteration {i+1}/20 with delta {delta_x/20}, {delta_y/20}", stack_info=True)
                    self._page.mouse.wheel(delta_x/20, delta_y/20)
                
                # Take a screenshot to return after the action
                return self.computer_screenshot()
            
            elif action == "wait":
                # For 'wait' action, we'll skip the waiting in async context
                # and just take a screenshot immediately
                logger.info("Processing wait action (non-blocking)", stack_info=True)
                
                # This effectively just takes a screenshot without waiting
                screenshot = self.computer_screenshot()
                return screenshot
                
            else:
                logger.warning(f"Unknown action: {action}", stack_info=True)
                # Take a screenshot anyway to avoid breaking the flow
                return self.computer_screenshot()
                
        except Exception as e:
            error_msg = f"Error performing computer action {action}: {e}"
            logger.error(error_msg, stack_info=True)
            logger.error(f"Action parameters: {kwargs}", stack_info=True)
            import traceback
            logger.error(f"Exception traceback: {traceback.format_exc()}", stack_info=True)
            response = ComputerResponse(action_result={"success": False, "error": error_msg})
            logger.info(f"BROWSER ACTION COMPLETED: {action}", stack_info=True)
            return response


class PlaywrightClient:
    """
    A client for local Playwright browser automation.
    This class mimics the Scrapybara API interface.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize a PlaywrightClient.
        
        Args:
            api_key: Not used for local Playwright, but kept for compatibility
        """
        self._playwright = None
        self._instances = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close all browser instances and the Playwright connection."""
        for instance in list(self._instances.values()):
            try:
                instance.close()
            except Exception as e:
                print(f"Error closing instance {instance.id}: {e}", stack_info=True)
        
        self._instances = {}
        
        if self._playwright:
            try:
                self._playwright.stop()
                self._playwright = None
            except Exception as e:
                print(f"Error stopping Playwright: {e}", stack_info=True)

    def get(self, instance_id: str) -> Optional[PlaywrightBrowserInstance]:
        """
        Get a browser instance by ID.
        
        Args:
            instance_id: The ID of the instance to get
            
        Returns:
            The browser instance, or None if not found
        """
        return self._instances.get(instance_id)

    def start_browser(
        self, 
        timeout_hours: float = 1.0,
        blocked_domains: Optional[List[str]] = None,
        browser_type: str = "chromium",
        headless: bool = False
    ) -> PlaywrightBrowserInstance:
        """
        Start a new browser instance.
        
        Args:
            timeout_hours: How long to keep the browser running (not strictly enforced)
            blocked_domains: List of domains to block (not implemented)
            browser_type: Type of browser to launch ('chromium', 'firefox', or 'webkit')
            headless: Whether to run the browser in headless mode
            
        Returns:
            A PlaywrightBrowserInstance for the new browser
        """
        # Initialize Playwright if not already initialized
        if not self._playwright:
            self._playwright = sync_playwright().start()
        
        # Select the browser based on type
        if browser_type == "firefox":
            browser_factory = self._playwright.firefox
        elif browser_type == "webkit":
            browser_factory = self._playwright.webkit
        else:
            browser_factory = self._playwright.chromium
            
        # Launch browser
        browser = browser_factory.launch(headless=headless)
        page = browser.new_page()
        page.goto("https://www.google.com")
        
        # Create a unique ID for this instance
        instance_id = str(uuid.uuid4())
        
        # Create and store the instance
        instance = PlaywrightBrowserInstance(instance_id, browser, page)
        self._instances[instance_id] = instance
        
        return instance


# For backward compatibility with Scrapybara, create aliases
class LocalPlaywright(PlaywrightClient):
    """Alias for PlaywrightClient for backward compatibility."""
    pass


# Compatibility aliases for Scrapybara types
BrowserInstance = PlaywrightBrowserInstance
UbuntuInstance = PlaywrightBrowserInstance  # Placeholder, not implemented
WindowsInstance = PlaywrightBrowserInstance  # Placeholder, not implemented
