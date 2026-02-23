import asyncio
import sys
import httpx
from bs4 import BeautifulSoup
import numpy as np
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer

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
    embedding: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


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
    client: httpx.AsyncClient, path: str, framework: str, embedding_model: SentenceTransformer
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

        # Grab the first substantial code block as the main snippet using a more specific selector
        code_snippet = ""
        # Target code blocks within pre tags, inside divs with class 'code-highlighting-enabled'
        main_code_block = soup.find("div", class_="code-highlighting-enabled")
        if main_code_block:
            pre_tag = main_code_block.find("pre")
            if pre_tag:
                code_tag = pre_tag.find("code")
                if code_tag:
                    code_snippet = code_tag.get_text(strip=True)

        # Look for dependency hints in specific installation code blocks
        dependencies = []
        install_headings = soup.find_all("h3")
        for heading in install_headings:
            heading_text = heading.get_text(strip=True).lower()
            if "install" in heading_text or "libraries" in heading_text:
                # Look for the next code block after an install-related heading
                next_code_block_div = heading.find_next_sibling(
                    "div", class_="language-bash code-highlighting-enabled"
                )
                if next_code_block_div:
                    code_tag = next_code_block_div.find("code")
                    if code_tag:
                        dependencies.append(code_tag.get_text(strip=True))

        # Derive category from URL: /docs/streamlit/tables/tables_read/ → "tables"
        parts = [p for p in path.strip("/").split("/") if p]
        category = parts[2] if len(parts) >= 3 else "general"

        # Generate embedding
        text_to_embed = f"{title} {description}"
        recipe_embedding = embedding_model.encode(text_to_embed, convert_to_numpy=True)

        return Recipe(
            title=title,
            description=description,
            url=f"{BASE_URL}{path}",
            framework=framework,
            category=category,
            code_snippet=code_snippet,
            dependencies=dependencies,
            embedding=recipe_embedding,
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

    # Initialize the embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    async with httpx.AsyncClient(timeout=15.0) as client:
        for entry in CATEGORY_URLS:
            framework = entry["framework"]
            category_path = entry["url"]

            print(f"[scraper] Collecting links for {framework}...", file=sys.stderr, flush=True)
            recipe_links = await collect_all_links(client, category_path)
            print(f"[scraper] Found {len(recipe_links)} recipe links for {framework}", file=sys.stderr, flush=True)

            # Scrape all recipe pages concurrently
            tasks = [
                scrape_recipe_page(client, link, framework, embedding_model)
                for link in recipe_links
            ]
            results = await asyncio.gather(*tasks)

            for recipe in results:
                if recipe:
                    recipes.append(recipe)

    return recipes