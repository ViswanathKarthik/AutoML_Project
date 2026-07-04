import pandas as pd
import numpy as np
from .utils import print_subsection


class DataPreprocessor:

    def __init__(self):

        self.preprocessing_summary = {
            "missing_values": {},
            "dropped_columns": {},
            "duplicates_removed": 0,
            "outliers": {}
        }
        # ✅ ADD THESE
        self.means = {}
        self.fill_values = {}
        self.outlier_bounds = {}
        self.drop_cols = []

    def load_data(self, filepath):

        df = pd.read_csv(filepath)

        print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        return df

    # -------------------------
    # Missing values
    # -------------------------

    def handle_missing(self, df):

        print_subsection("Handling Missing Values")

        df = df.copy()

        for col in df.columns:

            missing = df[col].isnull().sum()

            if missing == 0:
                continue

            pct = missing/len(df)*100
            
            if pct > 50:
                

                self.drop_cols.append(col)

                self.preprocessing_summary["dropped_columns"][col] = pct

                df.drop(columns=[col], inplace=True)

                print(f"  - Dropped {col} ({pct:.1f}% missing)")

                continue

            if df[col].dtype in ["int64","float64"]:

                mean_val = df[col].mean()

                # store it
                if not hasattr(self, "means"):
                    self.means = {}

                self.means[col] = mean_val

                df[col] = df[col].fillna(mean_val)

                strategy = "mean"

            else:
                if not hasattr(self, "fill_values"):
                    self.fill_values = {}

                self.fill_values[col] = "UNKNOWN"

                df[col] = df[col].fillna("UNKNOWN")

                strategy = "constant"

            self.preprocessing_summary["missing_values"][col] = {
                "count": int(missing),
                "percent": round(pct,2),
                "strategy": strategy
            }

            print(f"  - Filled {col} ({missing}) using {strategy}")

        return df

    # -------------------------
    # duplicates
    # -------------------------

    def remove_duplicates(self, df):

        print_subsection("Removing Duplicates")

        before = len(df)

        df = df.drop_duplicates()

        removed = before - len(df)

        self.preprocessing_summary["duplicates_removed"] = int(removed)

        print(f"✓ Removed {removed} duplicates")

        return df

    # -------------------------
    # outliers
    # -------------------------

    def handle_outliers(self, df):

        print_subsection("Handling Outliers")

        df = df.copy()

        numeric = df.select_dtypes(include=["int64","float64"]).columns

        for col in numeric:

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5*IQR
            upper = Q3 + 1.5*IQR

            outliers = ((df[col]<lower)|(df[col]>upper)).sum()

            if outliers > 0:

                self.preprocessing_summary["outliers"][col] = {
                    "count": int(outliers),
                    "lower": float(lower),
                    "upper": float(upper),
                    "mean_before": float(df[col].mean())
                }
                if not hasattr(self, "outlier_bounds"):
                    self.outlier_bounds = {}

                self.outlier_bounds[col] = (lower, upper)

                df[col] = np.clip(df[col], lower, upper)

                self.preprocessing_summary["outliers"][col]["mean_after"] = float(df[col].mean())

                print(f"  - capped {outliers} outliers in {col}")

        return df

    # -------------------------
    # preprocessing pipeline
    # -------------------------

    def preprocess(self, df, target):

        y = df[target]

        X = df.drop(columns=[target])

        X = self.handle_missing(X)

        X = self.remove_duplicates(X)

        X = self.handle_outliers(X)

        print_subsection("Preprocessing Complete")

        print(f"Final shape: {X.shape}")

        return X, y