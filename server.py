"""
LangGraph server for the CUA agent.
This enables running the agent with langgraph dev
"""

import os
from typing import Dict, List, Any, Optional

from fastapi import FastAPI
from langserve import add_routes
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph_cua import create_cua

# Create the FastAPI app
app = FastAPI(
    title="LangGraph CUA Agent",
    version="1.0",
    description="Computer User Agent using local Playwright",
)

# Create the CUA graph with local Playwright enabled
cua_graph = create_cua(
    use_local_playwright=True,  # Use local Playwright
    headless=False,  # Show the browser window
    browser_type="chromium"  # Use chromium (you could also use firefox or webkit)
)

# Define the input schema for the agent
@app.get("/")
async def root():
    """Root endpoint for health checks"""
    return {"status": "ok", "message": "CUA agent is running"}

# Add the LangServe routes for the agent
add_routes(
    app,
    cua_graph,
    path="/cua",
    input_type=Dict[str, List[Dict[str, str]]],  # Expecting {"messages": [{"role": "...", "content": "..."}]}
)

# This is for local development only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
