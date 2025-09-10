import os
import sys
import re
import pandas as pd
import json
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)
from src.feature_extraction.features_html import extract_html_features
from src.feature_extraction.features_css import extract_css_features
from src.preprocessing.data_processing import handle_missing_features, simplify_credibility_labels, encode_domain_type, remove_url_duplicates
from src.utils.helpers import get_domain_type

# Connecting files
def sanitize_filename_variants(url):
    url = url.rstrip('/')
    url = url.replace("https://", "").replace("http://", "")

    with_www = re.sub(r'[^a-zA-Z0-9\.]+', '', url)
    url_no_www = url.replace("www.", "")
    no_www = re.sub(r'[^a-zA-Z0-9\.]+', '', url_no_www)

    return [with_www, no_www]

# Load metadata
metadata_path = os.path.join(project_root, "data", "processed", "metadata_array1.json")
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

#metadata = metadata[:100]  # Limit to first 100 entries for testing

combined_data = []

for entry in metadata:
    if "source_url" not in entry:
        continue

    source_url = entry["source_url"]
    name_variants = sanitize_filename_variants(source_url)

    html_path = None
    css_path = None

    # Try all name variants until a file is found
    for name in name_variants:
        html_candidate = os.path.join(project_root, "web", "html", "html_front", f"{name}.html")
        css_candidate = os.path.join(project_root, "web", "css", "css_front", f"{name}.css")

        if not html_path and os.path.exists(html_candidate):
            html_path = html_candidate
            print(f"✅ Found HTML: {html_path}")

        if not css_path and os.path.exists(css_candidate):
            css_path = css_candidate
            print(f"✅ Found CSS: {css_path}")

        # If both are found, no need to keep looking
        if html_path and css_path:
            break

    # Only proceed if at least one type of content is found
    if html_path or css_path:
        html_features = extract_html_features(html_path) if html_path else {}
        css_features = extract_css_features(css_path) if css_path else {}

        combined_entry = {
            "url": source_url,
            #"bias_rating": entry.get("bias-rating"),
            "credibility_label": entry.get("mbfc-credibility-rating"),
            "media_type": entry.get("media-type"),
            "country": entry.get("country"),
            "traffic-popularity": entry.get("traffic-popularity"),
            **html_features,
            **css_features,
            "domain_type": get_domain_type(source_url) or "unknown",

        }
        combined_data.append(combined_entry)
    else:
        print(f"⚠️ Missing BOTH HTML and CSS for: {source_url}")


# After processing all entries
df = pd.DataFrame(combined_data)

df = handle_missing_features(df)

df = simplify_credibility_labels(df)

ngram_features = encode_domain_type(df)
df = pd.concat([df, ngram_features], axis=1)

df = remove_url_duplicates(df)

# Ensure directory exists and save CSV
os.makedirs(os.path.join(project_root, "data", "processed"), exist_ok=True)
df.to_csv(os.path.join(project_root, "data", "processed", "final_combined_dataset.csv"), index=False)

# Verify clearly your DataFrame is correct:
print(df.head())
