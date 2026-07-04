import pandas as pd


def feature_statistics(df):
    """
    Generate detailed statistics for each feature
    """

    stats = pd.DataFrame(index=df.columns)

    stats["missing_pct"] = (df.isnull().mean() * 100).round(2)
    stats["unique_values"] = df.nunique()

    numeric_cols = df.select_dtypes(include=["number"]).columns

    stats["mean"] = df[numeric_cols].mean()
    stats["median"] = df[numeric_cols].median()
    stats["min"] = df[numeric_cols].min()
    stats["max"] = df[numeric_cols].max()
    stats["std"] = df[numeric_cols].std()

    try:
        stats["mode"] = df.mode().iloc[0]
    except:
        stats["mode"] = None

    stats = stats.fillna("NA")

    return stats