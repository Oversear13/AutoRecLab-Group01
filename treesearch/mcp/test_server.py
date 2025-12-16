# Experimental test mcp server; to be removed before PR

from mcp.server.fastmcp import FastMCP

mcp = FastMCP()


@mcp.tool()
def greet(name: str):
    """Generates a greeting for a person with given name."""
    return f"Hello {name}, how are you feeling today?"


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
