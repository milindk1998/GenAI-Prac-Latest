from mcp.server.mcpserver import MCPServer

mcp = MCPServer("Calculator")

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

if __name__ == "__main__":
    print("Server is running........")
    mcp.run()
    