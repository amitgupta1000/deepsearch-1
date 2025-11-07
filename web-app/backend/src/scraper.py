#scraper.py
import logging, time, asyncio, os
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

# Try to import config with fallback
try:
    from .config import CACHE_ENABLED, CACHE_TTL
except ImportError:
    logging.warning("Could not import config. Using default cache settings.")
    CACHE_ENABLED = True
    CACHE_TTL = 3600


#=======================================================================================
# Define caching
class SimpleCache:
    def __init__(self, ttl: int = 3600):
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache and time.time() - self._timestamps.get(key, 0) < self._ttl:
            return self._cache[key]
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        return None

    def set(self, key: str, value: Any):
        self._cache[key] = value
        self._timestamps[key] = time.time()


# Initialize cache
# Ensure CACHE_ENABLED and CACHE_TTL are defined before this in your environment/setup
try:
    # Define a default TTL if CACHE_TTL is not found
    default_ttl = 3600 # Default to 1 hour if CACHE_TTL is not defined

    # Check if CACHE_ENABLED is defined, default to False if not
    cache_enabled = globals().get('CACHE_ENABLED', True)
    cache_ttl = globals().get('CACHE_TTL', default_ttl)


    if cache_enabled:
        cache = SimpleCache(ttl=cache_ttl)
        logging.info(f"Cache initialized with TTL={cache_ttl}s")
    else:
        cache = None # Cache disabled
        logging.info("Caching is disabled.")

except NameError as e:
    logging.warning(f"Cache related variable not found: {e}. Caching will be disabled.")
    cache = None
#=======================================================================================
# Define dataclass
@dataclass
class ScrapedContent:
    url: str
    title: str
    text: str
    html: Optional[str] = None
    content_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: Optional[str] = None
    scrape_time: float = 0.0

    def is_successful(self) -> bool:
        return self.success

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

#=======================================================================================
# main method
class Scraper:
    def __init__(self, cache_enabled: bool = CACHE_ENABLED, cache_ttl: int = CACHE_TTL):
        self.cache = SimpleCache(ttl=cache_ttl) if cache_enabled else None
        self._cache_enabled = cache_enabled


    async def _scrape_with_aiohttp(self, url: str) -> ScrapedContent:
        import aiohttp
        from bs4 import BeautifulSoup
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")
                    return ScrapedContent(
                        url=url,
                        title=soup.title.string if soup.title else url,
                        text=soup.get_text(separator="\n", strip=True),
                        html=html,
                        success=True,
                        scrape_time=time.time() - start
                    )
        except Exception as e:
            return ScrapedContent(url=url, success=False, error=str(e), scrape_time=time.time() - start)

    async def _scrape_with_requests_html(self, url: str) -> ScrapedContent:
        from requests_html import AsyncHTMLSession
        start = time.time()
        session = None
        try:
            session = AsyncHTMLSession()
            r = await session.get(url, timeout=30)
            text = r.html.full_text
            title = r.html.find("title", first=True).text if r.html.find("title", first=True) else url
            return ScrapedContent(
                url=url,
                title=title,
                text=text,
                html=r.html.html,
                success=True,
                scrape_time=time.time() - start
            )
        except Exception as e:
            return ScrapedContent(url=url, success=False, error=str(e), scrape_time=time.time() - start)
        finally:
            # Properly close the session to avoid warnings
            if session:
                try:
                    await session.close()
                except Exception as e:
                    logging.debug(f"Error closing session: {e}")

    async def _scrape_with_playwright(self, url: str) -> ScrapedContent:
        from playwright.async_api import async_playwright
        start = time.time()
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=30000)
                html = await page.content()
                text = await page.inner_text("body")
                title = await page.title()
                await browser.close()
                return ScrapedContent(
                    url=url,
                    title=title,
                    text=text,
                    html=html,
                    success=True,
                    scrape_time=time.time() - start
                )
        except Exception as e:
            return ScrapedContent(url=url, success=False, error=str(e), scrape_time=time.time() - start)

    async def _scrape_with_selenium(self, url: str) -> ScrapedContent:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from bs4 import BeautifulSoup
        start = time.time()
        try:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            driver = webdriver.Chrome(options=options)
            driver.get(url)
            time.sleep(2)
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            driver.quit()
            return ScrapedContent(
                url=url,
                title=soup.title.string if soup.title else url,
                text=soup.get_text(separator="\n", strip=True),
                html=html,
                success=True,
                scrape_time=time.time() - start
            )
        except Exception as e:
            return ScrapedContent(url=url, success=False, error=str(e), scrape_time=time.time() - start)

    async def scrape_url(self, url: str, dynamic: bool = False) -> ScrapedContent:
        logging.info(f"Scraping: {url}")

        # Check cache
        if self._cache_enabled and self.cache:
            cached = self.cache.get(url)
            if cached:
                try:
                    return ScrapedContent(**cached)
                except Exception as e:
                    logging.warning(f"Cache load failed: {e}")

        # Strategy list
        strategies = [
            self._scrape_with_aiohttp,
            self._scrape_with_playwright,
            self._scrape_with_selenium
        ]

        # Execute strategies
        for strategy in strategies:
            result = await strategy(url)
            logging.debug(f"{strategy.__name__} took {result.scrape_time:.2f}s")
            if result.is_successful():
                if self._cache_enabled and self.cache:
                    self.cache.set(url, result.to_dict())
                return result
            else:
                logging.warning(f"{strategy.__name__} failed: {result.error}")

        return ScrapedContent(url=url, success=False, error="All strategies failed.")
