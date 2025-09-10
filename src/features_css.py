import os
import re

def extract_css_features(css_path):
    with open(css_path, 'r', encoding='utf-8') as f:
        css_content = f.read()

    features = {
        "css_size_bytes": os.path.getsize(css_path),
        "num_selectors": css_content.count('{'),
        "num_classes": css_content.count('.'),
        "num_ids": css_content.count('#'),
        "num_media_queries": css_content.count('@media'),
    }

    # New features from literature
    features["has_flexbox"] = 'display:flex' in css_content.replace(" ", "").lower()
    features["has_grid"] = 'display:grid' in css_content.replace(" ", "").lower()
    features["has_position_absolute"] = 'position:absolute' in css_content.replace(" ", "").lower()
    features["uses_universal_selector"] = bool(re.search(r'[*]\s*{', css_content))
    features["uses_web_fonts"] = "@font-face" in css_content or "fonts.googleapis.com" in css_content
    features["has_bootstrap_classes"] = bool(re.search(r'\.(container|row|col-[a-z]+)', css_content))

    # Detect reset rules (common: * { margin: 0; padding: 0; })
    features["has_reset_rules"] = bool(
        re.search(r'\*\s*{[^}]*margin\s*:\s*0[^;]*;[^}]*padding\s*:\s*0[^;]*;', css_content)
    )

    # Existing feature calculations
    features["num_colors"] = len(set(re.findall(r'#[0-9a-fA-F]{3,6}', css_content)))
    features["num_font_weight_bold"] = css_content.count('font-weight:bold') + css_content.count('font-weight: bold')
    features["num_important_flags"] = css_content.count("!important")

    selectors = re.findall(r'([^{]+){', css_content)
    if selectors:
        avg_len = sum(len(s.strip()) for s in selectors) / len(selectors)
        features["avg_selector_length"] = round(avg_len, 2)
    else:
        features["avg_selector_length"] = 0

    features["num_imports"] = css_content.count("@import")

    fonts = re.findall(r'font-family\s*:\s*([^;]+);', css_content, re.IGNORECASE)
    font_set = set(f.strip().lower() for f in fonts)
    features["num_font_families"] = len(font_set)

    features["num_css_variables"] = css_content.count('--')
    features["num_animations"] = css_content.count("animation")
    features["num_transitions"] = css_content.count("transition")

    return features
