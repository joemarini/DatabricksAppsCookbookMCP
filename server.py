import asyncio
import sys
from mcp.server.fastmcp import FastMCP
from scraper import build_index, Recipe
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the MCP server
mcp = FastMCP("databricks-apps-cookbook")

# In-memory recipe index (populated at startup)
recipe_index: list[Recipe] = []

# Initialize the embedding model for queries
query_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def search_recipes(query: str, framework: str | None = None) -> list[Recipe]:
    """
    Performs a keyword-based search across recipe titles, descriptions, and categories.

    Args:
        query (str): The search query string.
        framework (str | None): Optional. Filters results by a specific framework (e.g., "Streamlit").

    Returns:
        list[Recipe]: A list of Recipe objects that match the keyword query,
                      optionally filtered by framework.
    """
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


def semantic_search_recipes(query: str, framework: str | None = None, top_k: int = 5) -> list[Recipe]:
    """
    Performs a semantic search across recipe embeddings to find the most relevant recipes.

    Args:
        query (str): The search query string.
        framework (str | None): Optional. Filters results by a specific framework (e.g., "Streamlit").
        top_k (int): The number of top similar recipes to return.

    Returns:
        list[Recipe]: A list of Recipe objects, semantically ranked by their similarity
                      to the query, optionally filtered by framework.
    """
    if not recipe_index:
        return []

    query_embedding = query_embedding_model.encode(query, convert_to_numpy=True)
    
    # Filter by framework first if specified
    candidates = [r for r in recipe_index if not framework or r.framework.lower() == framework.lower()]
    
    if not candidates:
        return []

    # Get embeddings of candidate recipes
    recipe_embeddings = np.array([r.embedding for r in candidates])

    # Calculate cosine similarities
    similarities = cosine_similarity([query_embedding], recipe_embeddings)[0]

    # Sort by similarity and get top_k
    ranked_indices = similarities.argsort()[::-1]
    
    # Return the actual Recipe objects, not just indices
    ranked_recipes = [candidates[i] for i in ranked_indices]
    return ranked_recipes[:top_k]


@mcp.tool()
async def search_recipes_tool(query: str, framework: str = "", semantic: bool = True) -> str:
    """
    Search the Databricks Apps Cookbook for recipes that match your need, supporting both
    semantic and keyword-based search.

    Args:
        query (str): A natural language query describing what you want to achieve,
                     e.g., 'read from a Delta table' or 'call an LLM'.
        framework (str): Optional. Filters the search results by a specific framework
                         like 'Streamlit', 'Dash', 'FastAPI', or 'Reflex'.
        semantic (bool): Optional. If True (default), performs a semantic search based on
                         meaning. If False, performs a keyword-based search.

    Returns:
        str: A formatted string containing the top 5 matching recipes (title, framework,
             description, and URL), separated by '---', or a message indicating no
             recipes were found or the index is still loading.
    """
    if not recipe_index:
        return "Recipe index is still loading. Please try again in a moment."

    if semantic:
        results = semantic_search_recipes(query, framework or None)
    else:
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
    Retrieves the full details, including code snippets and dependencies, for a specific recipe
    identified by its title.

    Args:
        title (str): The exact or partial title of the recipe to retrieve.
                     A case-insensitive partial match is performed.

    Returns:
        str: A formatted string containing the recipe's title, framework, URL, description,
             code snippet (if available, formatted as a Python code block), and dependencies.
             Returns a "not found" message if no matching recipe is located.
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
    Lists all available recipes in the Databricks Apps Cookbook, optionally filtered by framework.

    Args:
        framework (str): Optional. Filters the list to show only recipes associated with
                         a specific framework (e.g., 'Streamlit').

    Returns:
        str: A newline-separated string listing each recipe's title, framework, and URL.
             Returns a "No recipes found" message if the filtered list is empty.
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
    Re-scrapes the entire Databricks Apps Cookbook to refresh the in-memory recipe index.

    This function is useful for updating the recipe data if new content has been added
    to the cookbook since the server was last started or the index was built.

    Returns:
        str: A confirmation message indicating that the index has been refreshed
             and the total number of recipes loaded.
    """
    global recipe_index
    print("[server] Refreshing recipe index...", file=sys.stderr, flush=True)
    recipe_index = await build_index()
    return f"Index refreshed. {len(recipe_index)} recipes loaded."


async def main():
    """
    Main asynchronous function to initialize the server.

    This function is responsible for building the initial recipe index by
    calling the `build_index` function from the `scraper` module. It
    prints status messages to stderr during the indexing process.
    """
    global recipe_index
    print("[server] Building recipe index...", file=sys.stderr, flush=True)
    recipe_index = await build_index()
    print(f"[server] Indexed {len(recipe_index)} recipes.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
    mcp.run(transport="stdio")
