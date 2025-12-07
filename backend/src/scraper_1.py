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


    async def _scrape_with_aiohttp(self, url: str, retries: int = 2) -> ScrapedContent:
        import aiohttp
        from bs4 import BeautifulSoup
        from fake_useragent import UserAgent
        import traceback
        start = time.time()
        session = None
        last_exception = None
        # Use a realistic, modern User-Agent and browser-like headers
        realistic_user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        for attempt in range(retries + 1):
            try:
                headers = {
                    "User-Agent": realistic_user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "DNT": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1",
                    "Referer": "https://www.google.com/",
                }
                logging.info(f"[aiohttp] Attempt {attempt+1} for {url} with headers: {headers}")
                timeout = aiohttp.ClientTimeout(total=30)
                # Enable cookies for better compatibility
                session = aiohttp.ClientSession(headers=headers, timeout=timeout, cookie_jar=aiohttp.CookieJar())
                async with session.get(url) as response:
                    status = response.status
                    logging.info(f"[aiohttp] Response status: {status}")
                    logging.info(f"[aiohttp] Response headers: {dict(response.headers)}")
                    response.raise_for_status()
                    html = await response.text()
                    logging.info(f"[aiohttp] HTML length: {len(html)}")
                    soup = BeautifulSoup(html, "html.parser")
                    logging.info(f"[aiohttp] Successfully scraped {url} in {time.time() - start:.2f}s")
                    return ScrapedContent(
                        url=url,
                        title=soup.title.string if soup.title else url,
                        text=soup.get_text(separator="\n", strip=True),
                        html=html,
                        success=True,
                        scrape_time=time.time() - start
                    )
            except Exception as e:
                last_exception = e
                logging.error(f"[aiohttp] Attempt {attempt+1} failed for {url}: {e}")
                logging.error(traceback.format_exc())
                await asyncio.sleep(1)
            finally:
                if session is not None:
                    await session.close()
        logging.error(f"[aiohttp] All attempts failed for {url}. Last exception: {last_exception}")
        return ScrapedContent(
            url=url,
            title=url,
            text="",
            html=None,
            success=False,
            error=str(last_exception),
            scrape_time=time.time() - start
        )

    async def _scrape_with_requests_html(self, url: str, retries: int = 2) -> ScrapedContent:
        from requests_html import AsyncHTMLSession
        from fake_useragent import UserAgent
        import traceback
        start = time.time()
        session = None
        last_exception = None
        ua = UserAgent()
        for attempt in range(retries + 1):
            try:
                headers = {
                    "User-Agent": ua.random,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "DNT": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1"
                }
                logging.info(f"[requests_html] Attempt {attempt+1} for {url} with headers: {headers}")
                session = AsyncHTMLSession(browser_args=["--no-sandbox"])
                r = await session.get(url, timeout=30, headers=headers)
                logging.info(f"[requests_html] Response status: {r.status_code if hasattr(r, 'status_code') else 'N/A'}")
                logging.info(f"[requests_html] HTML length: {len(r.html.html) if r.html and r.html.html else 0}")
                text = r.html.full_text
                title = r.html.find("title", first=True).text if r.html.find("title", first=True) else url
                logging.info(f"[requests_html] Successfully scraped {url} in {time.time() - start:.2f}s")
                return ScrapedContent(
                    url=url,
                    title=title,
                    text=text,
                    html=r.html.html,
                    success=True,
                    scrape_time=time.time() - start
                )
            except Exception as e:
                last_exception = e
                logging.error(f"[requests_html] Attempt {attempt+1} failed for {url}: {e}")
                logging.error(traceback.format_exc())
                await asyncio.sleep(1)
            finally:
                if session:
                    try:
                        await session.close()
                    except Exception as e:
                        logging.debug(f"Error closing session: {e}")
        logging.error(f"[requests_html] All attempts failed for {url}. Last exception: {last_exception}")
        return ScrapedContent(url=url, success=False, error=str(last_exception), scrape_time=time.time() - start)

    async def _scrape_with_playwright(self, url: str) -> ScrapedContent:
        from playwright.async_api import async_playwright
        from fake_useragent import UserAgent
        import traceback
        start = time.time()
        ua = UserAgent()
        headers = {
            "User-Agent": ua.random,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "DNT": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1"
        }
        try:
            logging.info(f"[playwright] Launching browser for {url} with headers: {headers}")
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(user_agent=headers["User-Agent"], extra_http_headers=headers)
                page = await context.new_page()
                await page.goto(url, timeout=30000)
                logging.info(f"[playwright] Page loaded for {url}")
                html = await page.content()
                logging.info(f"[playwright] HTML length: {len(html)}")
                text = await page.inner_text("body")
                title = await page.title()
                await browser.close()
                logging.info(f"[playwright] Successfully scraped {url} in {time.time() - start:.2f}s")
                return ScrapedContent(
                    url=url,
                    title=title,
                    text=text,
                    html=html,
                    success=True,
                    scrape_time=time.time() - start
                )
        except Exception as e:
            logging.error(f"[playwright] Failed for {url}: {e}")
            logging.error(traceback.format_exc())
            return ScrapedContent(
                url=url,
                title=url,
                text="",
                html=None,
                success=False,
                error=str(e),
                scrape_time=time.time() - start
            )

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
        logging.info(f"[scraper] Starting scrape for: {url}")

        # Check cache
        if self._cache_enabled and self.cache:
            cached = self.cache.get(url)
            if cached:
                try:
                    logging.info(f"[scraper] Returning cached result for: {url}")
                    return ScrapedContent(**cached)
                except Exception as e:
                    logging.warning(f"[scraper] Cache load failed: {e}")

        # Use dynamic scraping for sites that likely need it, otherwise static first.
        use_dynamic_first = dynamic or any(
            site in url for site in ["reuters.com", "bloomberg.com", "wsj.com"]
        )

        if use_dynamic_first:
            logging.info(f"[scraper] Using dynamic-first scraping strategy for {url}")
            strategies = [self._scrape_with_playwright, self._scrape_with_aiohttp]
        else:
            logging.info(f"[scraper] Using static-first scraping strategy for {url}")
            strategies = [self._scrape_with_aiohttp, self._scrape_with_playwright]

        # Detailed logging of strategies
        logging.info(f"[scraper] Strategies to try: {[s.__name__ for s in strategies]}")

        # Execute strategies
        for idx, strategy in enumerate(strategies):
            logging.info(f"[scraper] Trying strategy {idx+1}/{len(strategies)}: {strategy.__name__}")
            if strategy is self._scrape_with_aiohttp:
                result = await strategy(url, retries=2)
            else:
                result = await strategy(url)

            logging.info(f"[scraper] {strategy.__name__} finished in {result.scrape_time:.2f}s, success={result.success}")
            if result.error:
                logging.warning(f"[scraper] {strategy.__name__} error: {result.error}")
            # Patch: Ensure ScrapedContent always has title and text
            if not getattr(result, "title", None):
                result.title = url
            if not getattr(result, "text", None):
                result.text = ""
            if result.is_successful():
                logging.info(f"[scraper] Strategy {strategy.__name__} succeeded for {url}")
                if self._cache_enabled and self.cache:
                    self.cache.set(url, result.to_dict())
                return result
            else:
                logging.info(f"[scraper] Strategy {strategy.__name__} failed for {url}, trying next...")
                logging.info("[scraper] Pausing for 1 second before next strategy...")
                await asyncio.sleep(1)

        logging.error(f"[scraper] All strategies failed for {url}")
        return ScrapedContent(url=url, title=url, text="", success=False, error="All strategies failed.")
