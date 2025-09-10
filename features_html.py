from bs4 import BeautifulSoup
import os

def extract_html_features(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
        soup = BeautifulSoup(html, 'html.parser')

    title_text = soup.title.string.strip() if soup.title and soup.title.string else ""

    text_length = len(soup.get_text(strip=True))
    html_size_bytes = os.path.getsize(html_path)

    features = {
        "html_size_bytes": html_size_bytes,
        "text_length": text_length,
        "title_length": len(title_text),
        "num_h1": len(soup.find_all('h1')),
        "num_headings_total": sum(len(soup.find_all(tag)) for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']),
        "has_meta_description": bool(soup.find('meta', attrs={"name": "description"})),
        "num_forms": len(soup.find_all('form')),
        "num_list_items": len(soup.find_all('li')),
        "has_lang_attr": 'lang' in soup.html.attrs if soup.html else False,
        "num_tables": len(soup.find_all('table')),
        "num_inline_styles": len(soup.find_all(style=True)),
        "num_noscript_tags": len(soup.find_all('noscript')),
        "num_comments": len(soup.find_all(string=lambda text: isinstance(text, str) and '<!--' in text)),
        "text_to_html_ratio": text_length / max(html_size_bytes, 1)
    }

    # Ad detection
    ad_keywords = ['ad', 'ads', 'sponsor', 'banner', 'promo', 'advertisement', 'adsbygoogle']

    def is_ad_element(tag):
        class_id = ' '.join(tag.get('class', []) + [tag.get('id', '')]).lower()
        return any(keyword in class_id for keyword in ad_keywords)

    features["num_ads_estimated"] = len(soup.find_all(is_ad_element)) + len(soup.find_all('iframe'))

    # === New features from Li et al. and DasGupta et al. ===
    features["num_iframes"] = len(soup.find_all('iframe'))
    features["num_scripts"] = len(soup.find_all('script'))
    features["num_meta_tags"] = len(soup.find_all('meta'))


    links = soup.find_all('a')
    features["num_links"] = len(links)

    internal_links = 0
    external_links = 0
    empty_links = 0

    for link in links:
        href = link.get("href", "").strip()
        if href in ["", "#", "javascript:void(0)", "javascript:;"]:
            empty_links += 1
        elif href.startswith("/") or not href.startswith("http"):
            internal_links += 1
        else:
            external_links += 1

    features["num_internal_links"] = internal_links
    features["num_external_links"] = external_links
    features["num_empty_links"] = empty_links
    features["link_internal_external_ratio"] = internal_links / max(external_links, 1)

    return features
