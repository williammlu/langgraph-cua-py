from langgraph_cua import create_cua
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create CUA with local Playwright
cua_graph = create_cua(
    use_local_playwright=True,  # Use local Playwright
    headless=False,  # Show the browser window
    browser_type="chromium"  # Use chromium (you could also use firefox or webkit)
)

# Define the input messages
messages = [
    {
        "role": "system",
        "content": (
            "You're an advanced AI computer use assistant. The browser you are using "
            "is already initialized, and visiting google.com."
        ),
    },
    {
        "role": "user",
        "content": (
            "Can you find the best price for new all season tires which will fit on my 2019 Subaru Forester?"
        ),
    },
]

async def main():
    # Run the CUA with the input messages
    print("Starting CUA execution")
    result = await cua_graph.ainvoke({"messages": messages})
    print(result)
    print("CUA execution completed")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())