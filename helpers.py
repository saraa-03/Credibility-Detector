from urllib.parse import urlparse

def get_domain_type(url):
    try:
        if not url.startswith("http"):
            url = "https://" + url  # force scheme for parsing
        netloc = urlparse(url).netloc
        domain_parts = netloc.split('.')
        if len(domain_parts) < 2:
            return "unknown"
        return domain_parts[-1].lower()
    except Exception:
        return "unknown"
