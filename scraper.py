import asyncio
import sys
import httpx
from bs4 import BeautifulSoup
from dataclasses import dataclass, field

BASE_URL = "https://apps-cookbook.dev"

CATEGORY_URLS = [
    {"framework": "Streamlit", "url": "/docs/category/streamlit"},
    {"framework": "Dash",      "url": "/docs/category/dash"},
    {"framework": "FastAPI",   "url": "/docs/category/fastapi"},
    {"framework": "Reflex",    "url": "/docs/category/reflex"},
]


@dataclass
class Recipe:
    title: str
    description: str
    url: str
    framework: str
    category: str
    code_snippet: str = ""
    dependencies: list[str] = field(default_factory=list)


async def fetch_page(client: httpx.AsyncClient, path: str) -> BeautifulSoup:
    response = await client.get(f"{BASE_URL}{path}", follow_redirects=True)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def is_recipe_url(href: str) -> bool:
    """
    Recipe pages are 3+ levels deep under /docs/ and are NOT category pages.
    e.g. /docs/streamlit/tables/tables_read/ — YES
         /docs/category/streamlit/           — NO
         /docs/intro                          — NO
         /docs/deploy                         — NO
    """
    if not href.startswith("/docs/"):
        return False
    if "/category/" in href:
        return False
    if href in ("/docs/intro", "/docs/intro/", "/docs/deploy", "/docs/deploy/"):
        return False
    # Must be at least 3 path segments deep: /docs/{framework}/{category}/{recipe}
    parts = [p for p in href.strip("/").split("/") if p]
    return len(parts) >= 3


async def scrape_recipe_page(
    client: httpx.AsyncClient, path: str, framework: str
) -> Recipe | None:
    try:
        soup = await fetch_page(client, path)

        title = soup.find("h1")
        title = title.get_text(strip=True) if title else ""
        if not title:
            return None

        article = soup.find("article")
        first_p = article.find("p") if article else None
        description = first_p.get_text(strip=True) if first_p else ""

        # Grab the first substantial code block as the main snippet
        code_snippet = ""
        for code in soup.find_all("code"):
            text = code.get_text(strip=True)
            if len(text) > 100:  # skip tiny inline code snippets
                code_snippet = text
                break

        # Look for dependency hints
        dependencies = []
        for code in soup.find_all("code"):
            text = code.get_text(strip=True)
            if text.startswith("pip install") or "requirements" in text.lower():
                dependencies.append(text)

        # Derive category from URL: /docs/streamlit/tables/tables_read/ → "tables"
        parts = [p for p in path.strip("/").split("/") if p]
        category = parts[2] if len(parts) >= 3 else "general"

        return Recipe(
            title=title,
            description=description,
            url=f"{BASE_URL}{path}",
            framework=framework,
            category=category,
            code_snippet=code_snippet,
            dependencies=dependencies,
        )
    except Exception as e:
        print(f"[scraper] Failed to scrape {path}: {e}", file=sys.stderr, flush=True)
        return None


async def collect_all_links(client: httpx.AsyncClient, start_path: str) -> set[str]:
    """
    Crawl a category page AND any sub-category pages to collect all recipe links.
    The cookbook nests recipes under sub-categories like /docs/category/streamlit/tables/
    so we need to follow those too.
    """
    visited = set()
    to_visit = {start_path}
    recipe_links = set()

    while to_visit:
        path = to_visit.pop()
        if path in visited:
            continue
        visited.add(path)

        try:
            soup = await fetch_page(client, path)
        except Exception as e:
            print(f"[scraper] Failed to fetch {path}: {e}", file=sys.stderr, flush=True)
            continue

        for a in soup.find_all("a", href=True):
            href = a["href"].split("?")[0].split("#")[0]  # strip query strings and anchors

            if is_recipe_url(href):
                recipe_links.add(href)
            elif "/category/" in href and href not in visited:
                # Follow sub-category pages to find more recipes
                to_visit.add(href)

    return recipe_links


async def build_index() -> list[Recipe]:
    recipes: list[Recipe] = []

    async with httpx.AsyncClient(timeout=15.0) as client:
        for entry in CATEGORY_URLS:
            framework = entry["framework"]
            category_path = entry["url"]

            print(f"[scraper] Collecting links for {framework}...", file=sys.stderr, flush=True)
            recipe_links = await collect_all_links(client, category_path)
            print(f"[scraper] Found {len(recipe_links)} recipe links for {framework}", file=sys.stderr, flush=True)

            # Scrape all recipe pages concurrently
            tasks = [
                scrape_recipe_page(client, link, framework)
                for link in recipe_links
            ]
            results = await asyncio.gather(*tasks)

            for recipe in results:
                if recipe:
                    recipes.append(recipe)

    return recipes