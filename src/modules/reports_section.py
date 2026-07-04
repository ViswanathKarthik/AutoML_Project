def feature_selection_section(log):

    lines = []

    lines.append("FEATURE SELECTION SUMMARY")
    lines.append("--------------------------------")

    lines.append(f"Initial Features: {len(log['initial_features'])}")

    sections = [
        "id_removed",
        "high_cardinality_removed",
        "constant_removed",
        "mi_removed",
        "correlation_removed"
    ]

    for s in sections:

        items = log.get(s, [])

        if len(items) > 0:

            lines.append(f"\n{s.replace('_',' ').title()} ({len(items)})")

            lines.append(", ".join(items))

    lines.append(f"\nFinal Features ({len(log['final_features'])})")

    lines.append(", ".join(log["final_features"]))

    return "\n".join(lines)


def dataset_overview(df, target):

    lines = []

    lines.append("DATASET OVERVIEW")
    lines.append("--------------------------------")

    lines.append(f"Rows: {df.shape[0]}")
    lines.append(f"Features: {df.shape[1]}")

    lines.append(f"Target Variable: {target}")

    return "\n".join(lines)