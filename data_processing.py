import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def load_json_data(filepath):
    return pd.read_json(filepath, lines=True)

def preprocess_df(df):
    df.drop(['bias-rating', 'factual-reporting'], axis=1, inplace=True)
    return df

def handle_missing_features(df):
    # Fill missing numeric features with 0
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Fill missing categorical/text features with 'unknown'
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        mode_val = df[col].mode()[0] if not df[col].mode().empty else 0
        df[col] = df[col].fillna(mode_val)
    return df


def simplify_credibility_labels(df):
    label_map = {
        "high credibility": 0,
        "medium credibility": 1,
        "low credibility": 1
    }

    # Normalize the labels: lowercase and strip whitespace
    df["credibility_label"] = df["credibility_label"].astype(str).str.strip().str.lower()

    # Map to scores
    df["credibility_score"] = df["credibility_label"].map(label_map)

    # Drop rows that couldn't be mapped
    df = df.dropna(subset=["credibility_score"]).copy()

    return df

def encode_domain_type(df, ngram_range=(3, 3), top_k=50):

    vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range, max_features=top_k)
    X_vec = vectorizer.fit_transform(df["url"])
    feature_df = pd.DataFrame(X_vec.toarray(), columns=vectorizer.get_feature_names_out())
    return feature_df


def remove_url_duplicates(df):
    return df.drop_duplicates(subset='url', keep='first').copy()
