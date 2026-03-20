import io

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader


def parse_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def parse_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)


async def fetch_url(url: str) -> str:
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "pdf" in content_type:
            return parse_pdf(response.content)
        return parse_html(response.text)


def chunk_text(
    text: str, chunk_size: int = 800, overlap: int = 200
) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(current.strip())
            words = current.split()
            overlap_words = max(1, overlap // 5)
            tail = " ".join(words[-overlap_words:]) if len(words) > overlap_words else ""
            current = tail + "\n\n" + para if tail else para
        else:
            current += ("\n\n" if current else "") + para

    if current.strip():
        chunks.append(current.strip())

    return chunks
