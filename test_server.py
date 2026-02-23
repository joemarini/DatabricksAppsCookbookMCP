import asyncio
import server
from fastmcp import Client

async def main():
    # Build the recipe index first
    await server.main()

    async with Client(server.mcp) as client:

        # List all available tools
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])
        print()

        # Test list_recipes
        result = await client.call_tool("list_recipes", {"framework": "Streamlit"})
        print("--- list_recipes (Streamlit) ---")
        print(result.data)
        print()

        # Test search
        result = await client.call_tool("search_recipes_tool", {
            "query": "embed a dashboard",
            "framework": "Reflex"
        })
        print("--- search_recipes_tool ---")
        print(result.data)
        print()

        # Test get_recipe
        result = await client.call_tool("get_recipe", {"title": "Read"})
        print("--- get_recipe ---")
        print(result.data)

asyncio.run(main())