from langchain_core.runnables.config import RunnableConfig

from ..types import CUAState
from ..utils import get_configuration_with_defaults, get_browser_client
from ..playwright_client import BrowserInstance

# List of domains to block for safety
BLOCKED_DOMAINS = [
    "maliciousbook.com",
    "evilvideos.com",
    "darkwebforum.com",
    "shadytok.com",
    "suspiciouspins.com",
    "ilanbigio.com",
]


def create_vm_instance(state: CUAState, config: RunnableConfig):
    """Create a local Playwright browser instance.
    
    Args:
        state: The current state of the CUA agent.
        config: The runnable configuration.
        
    Returns:
        A dictionary with the instance_id.
    """
    # Check if we already have an instance ID in the state
    instance_id = state.get("instance_id")
    if instance_id is not None:
        # If the instance_id already exists in state, do nothing.
        return {}
    
    # Get configuration
    configuration = get_configuration_with_defaults(config)
    timeout_hours = configuration.get("timeout_hours", 1.0)
    browser_type = configuration.get("browser_type", "chromium")
    headless = configuration.get("headless", False)
    
    # Get the browser client
    client = get_browser_client()
    
    # Process blocked domains
    blocked_domains = [
        domain.replace("https://", "").replace("www.", "") for domain in BLOCKED_DOMAINS
    ]
    
    # Start the browser
    instance = client.start_browser(
        timeout_hours=timeout_hours,
        blocked_domains=blocked_domains,
        browser_type=browser_type,
        headless=headless
    )
    
    # Return the instance ID
    return {
        "instance_id": instance.id,
    }
