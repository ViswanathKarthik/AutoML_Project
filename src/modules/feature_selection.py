"""
Smart Feature Selection Module
AutoML Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Tuple, Dict
from .utils import print_subsection
import config


class FeatureSelector:

    def __init__(
        self,
        mi_threshold=config.MI_THRESHOLD,
        correlation_threshold=config.CORRELATION_THRESHOLD,
        problem_type="classification",
        model_top_k=config.MODEL_TOP_K
    ):

        self.mi_threshold = mi_threshold
        self.correlation_threshold = correlation_threshold
        self.problem_type = problem_type
        self.model_top_k = model_top_k

        self.mi_scores = None
        self.selected_features = None
        self.encoders = {}

        self.selection_log = {
            "initial_features": [],
            "id_removed": [],
            "high_cardinality_removed": [],
            "constant_removed": [],
            "mi_removed": [],
            "correlation_removed": [],
            "model_selected": [],
            "final_features": [],
            "mi_scores": {}
        }

    # --------------------------------------------------
    # ID DETECTION
    # --------------------------------------------------

    def _is_id_name(self, col):

        keywords = ["id", "identifier", "account", "customer", "member", "transaction"]

        return any(k in col.lower() for k in keywords)

    def _is_sequential(self, series):

        if not pd.api.types.is_numeric_dtype(series):
            return False

        diff = series.diff().dropna()

        return diff.nunique() == 1

    # --------------------------------------------------
    # DETECT FEATURE TYPES
    # --------------------------------------------------

    def detect_feature_types(self, X):

        numeric_cols = []
        categorical_cols = []
        high_cardinality_cols = []
        id_cols = []

        for col in X.columns:

            series = X[col]

            if self._is_id_name(col):
                id_cols.append(col)
                continue

            if self._is_sequential(series):
                id_cols.append(col)
                continue

            if pd.api.types.is_numeric_dtype(series):
                numeric_cols.append(col)
                continue

            unique_ratio = series.nunique() / len(series)

            if unique_ratio > 0.98:
                high_cardinality_cols.append(col)

            elif series.nunique() < config.CATEGORICAL_THRESHOLD:
                categorical_cols.append(col)

            else:
                high_cardinality_cols.append(col)

        return numeric_cols, categorical_cols, high_cardinality_cols, id_cols

    # --------------------------------------------------
    # REMOVE CONSTANT FEATURES
    # --------------------------------------------------

    def remove_constant_features(self, X):

        print_subsection("Removing Constant Features")

        constant_cols = [c for c in X.columns if X[c].nunique() <= 1]

        for col in constant_cols:
            print(f"  ✗ {col}")

        self.selection_log["constant_removed"] = constant_cols

        return X.drop(columns=constant_cols, errors="ignore")

    # --------------------------------------------------
    # REMOVE ID / HIGH CARDINALITY
    # --------------------------------------------------

    def remove_high_cardinality(self, X, id_cols, high_card_cols):

        print_subsection("Removing ID / High Cardinality")

        drop_cols = id_cols + high_card_cols

        for col in drop_cols:
            print(f"  ✗ {col}")

        self.selection_log["id_removed"] = id_cols
        self.selection_log["high_cardinality_removed"] = high_card_cols

        return X.drop(columns=drop_cols, errors="ignore")

    # --------------------------------------------------
    # ENCODE CATEGORICAL
    # --------------------------------------------------

    def encode_categorical(self, X, categorical_cols):

        print_subsection("Encoding Categorical Features")

        categorical_cols = [c for c in categorical_cols if c in X.columns]

        for col in categorical_cols:

            print(f"  ✓ Encoding {col}")

            X[col] = X[col].astype(str)

            categories = sorted(X[col].unique().tolist())

            mapping = {cat: i for i, cat in enumerate(categories)}

            self.encoders[col] = mapping

            X[col] = X[col].map(mapping)

        return X

    # --------------------------------------------------
    # MUTUAL INFORMATION
    # --------------------------------------------------

    def calculate_mutual_information(self, X, y):

        print_subsection("Calculating Mutual Information")

        if self.problem_type == "classification":

            mi_scores = mutual_info_classif(
                X,
                y,
                random_state=config.RANDOM_STATE
            )

        else:

            mi_scores = mutual_info_regression(
                X,
                y,
                random_state=config.RANDOM_STATE
            )

        mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

        self.mi_scores = mi_series

        self.selection_log["mi_scores"] = mi_series.to_dict()

        print("\nTop MI Features")

        for i, (feat, score) in enumerate(mi_series.head(10).items(), 1):
            print(f"  {i}. {feat:35s} {score:.4f}")

        return mi_series

    # --------------------------------------------------
    # MI FILTER
    # --------------------------------------------------

    def filter_by_mi(self, X, y):

        print_subsection("Filtering by MI")

        mi_scores = self.calculate_mutual_information(X, y)

        keep_features = mi_scores[mi_scores >= self.mi_threshold].index

        removed = list(set(X.columns) - set(keep_features))

        for col in removed:
            print(f"  ✗ {col} (low MI)")

        self.selection_log["mi_removed"] = removed

        return X[keep_features]

    # --------------------------------------------------
    # CORRELATION FILTER
    # --------------------------------------------------

    def drop_correlated_features(self, X):

        print_subsection("Removing Correlated Features")

        corr_matrix = X.corr().abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        features_to_drop = []

        for column in upper.columns:

            high_corr = upper[column][upper[column] > self.correlation_threshold]

            for idx in high_corr.index:

                if self.mi_scores[column] >= self.mi_scores[idx]:
                    features_to_drop.append(idx)
                else:
                    features_to_drop.append(column)

        features_to_drop = list(set(features_to_drop))

        for col in features_to_drop:
            print(f"  ✗ {col}")

        self.selection_log["correlation_removed"] = features_to_drop

        return X.drop(columns=features_to_drop, errors="ignore")

    # --------------------------------------------------
    # MODEL BASED SELECTION
    # --------------------------------------------------

    def model_based_selection(self, X, y):

        print_subsection("Model Based Feature Selection")

        if self.problem_type == "classification":

            model = RandomForestClassifier(
                n_estimators=200,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )

        else:

            model = RandomForestRegressor(
                n_estimators=200,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )

        model.fit(X, y)

        importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        print("\nTop Model Features")

        for i, (feat, val) in enumerate(importance.head(10).items(), 1):
            print(f"  {i}. {feat:35s} {val:.4f}")

        top_features = importance.head(self.model_top_k).index.tolist()

        self.selection_log["model_selected"] = top_features

        return X[top_features]

    # --------------------------------------------------
    # MAIN PIPELINE
    # --------------------------------------------------

    def select_features(self, X, y) -> Tuple[pd.DataFrame, Dict]:

        print_subsection("Starting Smart Feature Selection")

        self.selection_log["initial_features"] = list(X.columns)

        print(f"Initial Features: {len(X.columns)}")

        num_cols, cat_cols, high_card, ids = self.detect_feature_types(X)

        X = self.remove_constant_features(X)

        X = self.remove_high_cardinality(X, ids, high_card)

        X = self.encode_categorical(X, cat_cols)

        X = self.filter_by_mi(X, y)

        X = self.drop_correlated_features(X)

        X = self.model_based_selection(X, y)

        self.selected_features = X.columns.tolist()

        self.selection_log["final_features"] = self.selected_features

        print_subsection("Feature Selection Complete")

        print(f"Final Features: {len(self.selected_features)}")



        return X, self.selection_log
    
    def transform(self, X):

        X = X.copy()

        # Drop removed columns
        cols_to_drop = (
            self.selection_log["id_removed"] +
            self.selection_log["high_cardinality_removed"] +
            self.selection_log["constant_removed"]
        )

        X = X.drop(columns=cols_to_drop, errors="ignore")

        # Apply stored encoding
        for col, mapping in self.encoders.items():

            if col in X.columns:

                X[col] = X[col].astype(str)

                X[col] = X[col].map(mapping).fillna(-1)

        # Ensure final selected columns
        for col in self.selected_features:
            if col not in X.columns:
                X[col] = 0

        return X[self.selected_features]