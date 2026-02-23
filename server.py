import asyncio
import sys
from mcp.server.fastmcp import FastMCP
from scraper import build_index, Recipe

# Initialize the MCP server
mcp = FastMCP("databricks-apps-cookbook")

# In-memory recipe index (populated at startup)
recipe_index: list[Recipe] = []


def search_recipes(query: str, framework: str | None = None) -> list[Recipe]:
    """Keyword search across title, description, and category."""
    terms = query.lower().split()

    return [
        r for r in recipe_index
        if all(
            term in f"{r.title} {r.description} {r.category}".lower()
            for term in terms
        )
        and (
            not framework
            or r.framework.lower() == framework.lower()
        )
    ]


@mcp.tool()
async def search_recipes_tool(query: str, framework: str = "") -> str:
    """
    Search the Databricks Apps Cookbook for recipes that match your need.

    Args:
        query: What you want to do, e.g. 'read from a Delta table' or 'call an LLM'
        framework: (Optional) Filter by framework: Streamlit, Dash, FastAPI, or Reflex
    """
    if not recipe_index:
        return "Recipe index is still loading. Please try again in a moment."

    results = search_recipes(query, framework or None)

    if not results:
        return f'No recipes found for "{query}". Try broader terms.'

    lines = []
    for r in results[:5]:
        lines.append(f"**{r.title}** ({r.framework})")
        lines.append(r.description)
        lines.append(f"🔗 {r.url}")
        lines.append("---")

    return "\n".join(lines)


@mcp.tool()
async def get_recipe(title: str) -> str:
    """
    Get the full code snippet and details for a specific recipe by title.

    Args:
        title: The exact or partial title of the recipe to retrieve
    """
    match = next(
        (r for r in recipe_index if title.lower() in r.title.lower()),
        None,
    )

    if not match:
        return f'Recipe "{title}" not found. Try list_recipes to see all available recipes.'

    parts = [
        f"# {match.title}",
        f"**Framework:** {match.framework}",
        f"**URL:** {match.url}",
        f"\n{match.description}",
    ]

    if match.code_snippet:
        parts.append(f"\n```python\n{match.code_snippet}\n```")

    if match.dependencies:
        parts.append("\n**Dependencies:**")
        parts.extend(match.dependencies)

    return "\n".join(parts)


@mcp.tool()
async def list_recipes(framework: str = "") -> str:
    """
    List all available recipes in the cookbook.

    Args:
        framework: (Optional) Filter by framework: Streamlit, Dash, FastAPI, or Reflex
    """
    filtered = (
        [r for r in recipe_index if r.framework.lower() == framework.lower()]
        if framework
        else recipe_index
    )

    if not filtered:
        return "No recipes found."

    lines = [f"- **{r.title}** ({r.framework}): {r.url}" for r in filtered]
    return "\n".join(lines)


@mcp.tool()
async def refresh_index() -> str:
    """
    Re-scrape the Databricks Apps Cookbook and refresh the recipe index.
    Useful if new recipes have been added since the server started.
    """
    global recipe_index
    recipe_index = await build_index()
    return f"Index refreshed. {len(recipe_index)} recipes loaded."


async def main():
    global recipe_index
    print("[server] Building recipe index...", file=sys.stderr, flush=True)
    recipe_index = await build_index()
    print(f"[server] Indexed {len(recipe_index)} recipes.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
    mcp.run(transport="stdio")
