"""Read a bounded, application-owned set of public cultural references."""
import asyncio
from html.parser import HTMLParser

import httpx

REFERENCES = (
    ("Museum Negeri NTB — Wayang Sasak", "https://museumnegeri.ntbprov.go.id/index.php/tulisan/29"),
    ("Yayasan Pedalangan Wayang Sasak — Tradisi pedalangan", "https://wayangsasak.org/baca/berita/26/Multi-Talenta%2C-Kini-Bertahan-Hidup-dari-Jual-Wayang"),
    ("Yayasan Pedalangan Wayang Sasak — Wong Menak Jayeng Rane", "https://wayangsasak.org/baca/tokoh/3/Wong-Menak-Jayeng-Rane-"),
    ("Museum Wayang Sasak — Nengah Gowang", "https://museum.wayangsasak.org/nengah-gowang-legenda-dalang-wayang-sasak/"),
)


class PageText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav", "header", "footer", "noscript"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav", "header", "footer", "noscript"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(" ".join(data.split()))


async def fetch_references():
    async with httpx.AsyncClient(timeout=7, follow_redirects=False) as client:
        async def fetch(index, title, url):
            try:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    if "text/html" not in response.headers.get("content-type", ""):
                        return None
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > 1_000_000:
                            return None
                parser = PageText()
                parser.feed(body.decode("utf-8", errors="replace"))
                text = "\n".join(parser.parts)
                if "sasak" not in text.casefold() or len(text) < 150:
                    return None
                return {"source_id": index, "title": title, "url": url, "text": text[:12000]}
            except (httpx.HTTPError, ValueError):
                return None
        rows = await asyncio.gather(*(fetch(i, *ref) for i, ref in enumerate(REFERENCES)))
    return [row for row in rows if row is not None]
