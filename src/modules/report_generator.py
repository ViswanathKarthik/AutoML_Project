import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from reportlab.platypus import PageBreak, Spacer, Paragraph
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Preformatted, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    PageBreak
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

from sklearn.metrics import confusion_matrix, roc_curve, auc


class AutoMLReport:

    def __init__(self, output_dir="report_assets"):

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.styles = getSampleStyleSheet()

    # --------------------------------------------------
    # EDA HISTOGRAMS
    # --------------------------------------------------

    def feature_histograms(self, df):

        paths = []

        numeric = df.select_dtypes(include=["int64","float64"])

        for col in numeric.columns:

            plt.figure(figsize=(6,4))
            sns.histplot(df[col], kde=True)

            plt.title(f"Distribution of {col}")

            path = f"{self.output_dir}/hist_{col}.png"

            plt.tight_layout()
            plt.savefig(path)
            plt.close()

            paths.append(path)

        return paths


    # --------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------

    def target_plot(self, df, target):

        plt.figure()

        sns.countplot(x=df[target])

        plt.title("Target Distribution")

        path = f"{self.output_dir}/target_distribution.png"

        plt.savefig(path)
        plt.close()

        return path


    # --------------------------------------------------
    # FEATURE VS TARGET
    # --------------------------------------------------

    def feature_vs_target(self, df, target):

        paths = []

        numeric = df.select_dtypes(include=["int64","float64"])

        for col in numeric.columns:

            if col == target:
                continue

            plt.figure(figsize=(6,4))
            sns.boxplot(x=df[target], y=df[col])

            plt.title(f"{col} vs {target}")

            path = f"{self.output_dir}/impact_{col}.png"

            plt.savefig(path)
            plt.close()

            paths.append(path)

        return paths


    # --------------------------------------------------
    # CORRELATION HEATMAP
    # --------------------------------------------------

    def correlation_heatmap(self, df):

        numeric = df.select_dtypes(include=["int64","float64"])

        plt.figure(figsize=(8,6))

        sns.heatmap(numeric.corr(), cmap="coolwarm")

        plt.title("Feature Correlation Heatmap")

        path = f"{self.output_dir}/correlation_heatmap.png"

        plt.savefig(path)
        plt.close()

        return path


    # --------------------------------------------------
    # OUTLIER BOXPLOTS
    # --------------------------------------------------

    def outlier_boxplots(self, df):

        paths = []

        numeric = df.select_dtypes(include=["int64","float64"])

        for col in numeric.columns:

            plt.figure()

            sns.boxplot(x=df[col])

            plt.title(f"Outliers in {col}")

            path = f"{self.output_dir}/outlier_{col}.png"

            plt.savefig(path)
            plt.close()

            paths.append(path)

        return paths


    # --------------------------------------------------
    # DISTRIBUTION AFTER PREPROCESSING
    # --------------------------------------------------

    def post_preprocessing_plots(self, X):

        paths = []

        for col in X.select_dtypes(include=["number"]).columns:

            plt.figure()

            sns.histplot(X[col], kde=True)

            plt.title(f"Post-Processing Distribution: {col}")

            path = f"{self.output_dir}/post_{col}.png"

            plt.savefig(path)
            plt.close()

            paths.append(path)

        return paths


    # --------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------

    def feature_importance_plot(self, importance_df):

        if importance_df is None:
            return None

        plt.figure(figsize=(8,5))

        sns.barplot(
            x="importance",
            y="feature",
            data=importance_df.head(10)
        )

        plt.title("Top Feature Importance")

        path = f"{self.output_dir}/feature_importance.png"

        plt.savefig(path)
        plt.close()

        return path


    # --------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------

    def roc_plot(self, model, X_test, y_test, name):

        if not hasattr(model, "predict_proba"):
            return None

        y_prob = model.predict_proba(X_test)[:,1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        roc_auc = auc(fpr, tpr)

        plt.figure()

        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1],[0,1],"--")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.title(name)

        plt.legend()

        path = f"{self.output_dir}/roc_{name}.png"

        plt.savefig(path)
        plt.close()

        return path


    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    def confusion_plot(self, model, X_test, y_test, name):

        preds = model.predict(X_test)

        cm = confusion_matrix(y_test, preds)

        plt.figure()

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

        plt.title(name)

        path = f"{self.output_dir}/cm_{name}.png"

        plt.savefig(path)
        plt.close()

        return path


    # --------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------

    def model_comparison_chart(self, results):

        df = pd.DataFrame(results).T

        plt.figure()

        df["F1-Score"].plot(kind="bar")

        plt.title("Model Comparison (F1 Score)")

        path = f"{self.output_dir}/model_comparison.png"

        plt.savefig(path)
        plt.close()

        return path


    # --------------------------------------------------
    # PDF GENERATOR
    # --------------------------------------------------

    def generate_pdf(
        self,
        filepath,
        dataset,
        target,
        models,
        results,
        best_model,
        preprocessing_log,
        feature_selection_log,
        feature_importance,
        X_preprocessed,
        X_test,
        y_test,
        pipeline_logs=None
    ):

        elements = []
        styles = self.styles

        # -----------------------------------------
        # TITLE
        # -----------------------------------------

        elements.append(Paragraph("AutoML Model Audit Report", styles["Title"]))
        elements.append(Spacer(1,20))

        elements.append(
            Paragraph(
                "This document explains every step taken by the AutoML pipeline "
                "from raw dataset to final model selection.",
                styles["Normal"]
            )
        )

        elements.append(PageBreak())

        # -----------------------------------------
        # DATASET OVERVIEW
        # -----------------------------------------
        elements.append(Paragraph("Feature Statistics (Before Preprocessing)", styles["Heading1"]))

        stats_table = self.feature_statistics(dataset)

        elements.append(Table(stats_table))

        elements.append(PageBreak())

        elements.append(Paragraph("Dataset Overview", styles["Heading1"]))

        elements.append(
            Paragraph(
                f"Rows: {dataset.shape[0]} | Columns: {dataset.shape[1]}",
                styles["Normal"]
            )
        )

        target_plot = self.target_plot(dataset, target)

        elements.append(Image(target_plot, width=400, height=250))

        elements.append(PageBreak())

        # -----------------------------------------
        # EDA
        # -----------------------------------------

        elements.append(Paragraph("Exploratory Data Analysis", styles["Heading1"]))

        hist = self.feature_histograms(dataset)

        for p in hist[:6]:
            elements.append(Image(p, width=400, height=250))

        corr = self.correlation_heatmap(dataset)

        elements.append(Image(corr, width=400, height=300))

        elements.append(PageBreak())

        # -----------------------------------------
        # OUTLIERS
        # -----------------------------------------

        elements.append(Paragraph("Outlier Detection", styles["Heading1"]))

        outliers = self.outlier_boxplots(dataset)

        for p in outliers[:6]:
            elements.append(Image(p, width=400, height=250))

        elements.append(PageBreak())

        # -----------------------------------------
        # FEATURE VS TARGET
        # -----------------------------------------

        elements.append(Paragraph("Feature Impact on Target", styles["Heading1"]))

        impacts = self.feature_vs_target(dataset, target)

        for p in impacts[:6]:
            elements.append(Image(p, width=400, height=250))

        elements.append(PageBreak())

        # -----------------------------------------
        # PREPROCESSING SUMMARY
        # -----------------------------------------

        elements.append(Paragraph("Preprocessing Summary", styles["Heading1"]))

        table = [["Feature","Missing Count","Missing %","Strategy"]]

        for col,info in preprocessing_log["missing_values"].items():

            table.append([
                col,
                info["count"],
                info["percent"],
                info["strategy"]
            ])

        elements.append(Table(table))

        elements.append(
            Paragraph(
                f"Duplicate rows removed: {preprocessing_log['duplicates_removed']}",
                styles["Normal"]
            )
        )

        elements.append(PageBreak())

        # -----------------------------------------
        # POST PREPROCESSING DISTRIBUTIONS
        # -----------------------------------------
        elements.append(Paragraph("Feature Statistics (After Preprocessing)", styles["Heading1"]))

        post_stats = self.feature_statistics(X_preprocessed)

        elements.append(Table(post_stats))

        elements.append(PageBreak())

        elements.append(Paragraph("Distributions After Preprocessing", styles["Heading1"]))

        post = self.post_preprocessing_plots(X_preprocessed)

        for p in post[:6]:
            elements.append(Image(p, width=400, height=250))

        elements.append(PageBreak())

        # -----------------------------------------
        # FEATURE SELECTION
        # -----------------------------------------

        elements.append(Paragraph("Feature Selection", styles["Heading1"]))

        elements.append(
            Paragraph(
                f"Initial features: {len(feature_selection_log['initial_features'])}",
                styles["Normal"]
            )
        )

        # elements.append(
        #     Paragraph(
        #         f"Removed features: {len(feature_selection_log['removed_features'])}",
        #         styles["Normal"]
        #     )
        # )

        elements.append(
            Paragraph(
                f"Final features: {len(feature_selection_log['final_features'])}",
                styles["Normal"]
            )
        )

        table = [["Final Features Used"]]

        for f in feature_selection_log["final_features"]:
            table.append([f])

        elements.append(Table(table))

        elements.append(PageBreak())

        # -----------------------------------------
        # FEATURE IMPORTANCE
        # -----------------------------------------

        elements.append(Paragraph("Feature Importance", styles["Heading1"]))

        imp_plot = self.feature_importance_plot(feature_importance)

        if imp_plot:
            elements.append(Image(imp_plot, width=400, height=300))

        elements.append(PageBreak())

        # -----------------------------------------
        # MODEL COMPARISON
        # -----------------------------------------

        elements.append(Paragraph("Model Comparison", styles["Heading1"]))

        chart = self.model_comparison_chart(results)

        elements.append(Image(chart, width=400, height=250))

        elements.append(PageBreak())

        # -----------------------------------------
        # INDIVIDUAL MODEL PERFORMANCE
        # -----------------------------------------

        elements.append(Paragraph("Model Evaluation", styles["Heading1"]))

        for name, model in models.items():

            elements.append(Paragraph(name, styles["Heading2"]))

            roc = self.roc_plot(model, X_test, y_test, name)

            cm = self.confusion_plot(model, X_test, y_test, name)

            if roc:
                elements.append(Image(roc, width=400, height=250))

            elements.append(Image(cm, width=400, height=250))

        elements.append(PageBreak())

        # -----------------------------------------
        # BEST MODEL
        # -----------------------------------------

        elements.append(Paragraph("Best Model Selected", styles["Heading1"]))

        elements.append(
            Paragraph(
                f"The AutoML pipeline selected <b>{best_model}</b> as the best performing model.",
                styles["Normal"]
            )
        )
        

        # -----------------------------------------
        # PIPELINE SUMMARY
        # -----------------------------------------

        elements.append(Paragraph("Pipeline Summary", styles["Heading1"]))

        summary = [
        ["Step","Description"],
        ["Data Loading","CSV loaded into pandas"],
        ["Preprocessing","Missing values handled, duplicates removed, outliers capped"],
        ["Feature Selection","MI filter + correlation removal + RF importance"],
        ["Model Training","Multiple models trained and evaluated"],
        ["Best Model Selection","Model with best F1 / R2 selected"],
        ["Model Explainability","SHAP used for feature importance"]
        ]

        elements.append(Table(summary))

        # -----------------------------------------
        # BUILD PDF
        # -----------------------------------------
        

        if pipeline_logs:

            elements.append(PageBreak())
            elements.append(Paragraph("Appendix: Execution Logs", styles["Heading1"]))
            elements.append(Spacer(1, 12))

            log_style = ParagraphStyle(
                name="LogStyle",
                fontName="Courier",
                fontSize=7.5,
                leading=9,
                leftIndent=6,
                rightIndent=6,
                borderPadding=6,
                backColor=colors.whitesmoke
            )

            elements.append(
                Preformatted(
                    pipeline_logs,
                    log_style
                )
            )

        
        doc = SimpleDocTemplate(filepath, pagesize=letter)

        doc.build(elements)

        print(f"\n AutoML report generated: {filepath}")
    def feature_statistics(self, df):

        stats = []

        for col in df.columns:

            stats.append([
                col,
                str(df[col].dtype),
                df[col].nunique(),
                round(df[col].mean(),3) if pd.api.types.is_numeric_dtype(df[col]) else "",
                round(df[col].std(),3) if pd.api.types.is_numeric_dtype(df[col]) else "",
                round(df[col].min(),3) if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]) else "",
                round(df[col].max(),3) if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]) else ""
            ])

        table = [["Feature","Type","Unique","Mean","Std","Min","Max"]] + stats

        return table
    def missing_value_table(self, df):

        rows = [["Feature","Missing Count","Missing %"]]

        for col in df.columns:

            miss = df[col].isnull().sum()

            if miss > 0:

                pct = round(miss/len(df)*100,2)

                rows.append([col, miss, pct])

        return rows