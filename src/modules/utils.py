import pandas as pd

def is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)

def is_categorical(series):
    return series.dtype == "object"

def print_section(title):
    print("\n" + "="*60)
    print(title)
    print("="*60)

def print_subsection(title):
    print("\n" + title)
    print("-"*len(title))

def detect_problem_type(y):
    if y.nunique() <= 10:
        return "classification"
    return "regression"