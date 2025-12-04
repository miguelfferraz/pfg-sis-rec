import json
from pathlib import Path
from typing import Dict, List


class LatexTableGenerator:
    MODEL_NAME_MAP = {
        "baseline": "Baseline",
        "knn_basic": "k-NN",
        "knn_baseline": "k-NN Baseline",
        "knn_means": "k-NN Means",
        "knn_zscore": "k-NN Z-Score",
        "svd": "SVD",
        "svdpp": "SVD++",
        "nmf": "NMF",
    }

    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.training_results = self._load_training_results()
        self.fairness_results = self._load_fairness_results()

    def _load_training_results(self) -> Dict:
        training_file = self.results_path / "training_results.json"
        if not training_file.exists():
            raise FileNotFoundError(f"Training results not found: {training_file}")

        with open(training_file, "r") as f:
            return json.load(f)

    def _load_fairness_results(self) -> Dict:
        fairness_file = self.results_path / "fairness_results.json"
        if not fairness_file.exists():
            return {}

        with open(fairness_file, "r") as f:
            return json.load(f)

    def generate_accuracy_table(self, metrics: List[str] = None) -> str:
        """
        Generate LaTeX table for model accuracy metrics.

        Args:
            metrics: List of metrics to include. Default: ["rmse", "mae", "fcp"]

        Returns:
            LaTeX table string
        """
        if metrics is None:
            metrics = ["rmse", "mae", "fcp"]

        # Build header
        metric_headers = " & ".join([m.upper() for m in metrics])
        header = f"Modelo & {metric_headers} & Tempo de Execução (s) \\\\"

        # Build rows
        rows = []
        for model_name, results in self.training_results.items():
            model_display = self.MODEL_NAME_MAP.get(model_name, model_name.title())

            # Get metrics
            metric_values = []
            for metric in metrics:
                if "mean_metrics" in results and metric in results["mean_metrics"]:
                    value = results["mean_metrics"][metric]
                    metric_values.append(f"{value:.4f}")
                elif "metrics" in results and metric in results["metrics"]:
                    value = results["metrics"][metric]
                    metric_values.append(f"{value:.4f}")
                else:
                    metric_values.append("--")

            # Get training time
            training_time = results.get("training_time", 0)
            metric_values.append(f"{training_time:.2f}")

            row = f"{model_display} & " + " & ".join(metric_values) + " \\\\"
            rows.append(row)

        # Build complete table
        col_format = "|l|" + "c " * len(metrics) + "c|"

        table = f"""\\begin{{center}}
\\begin{{tabular}}{{ {col_format} }}
    \\hline
    {header}
    \\hline
    {chr(10).join(f'    {row}' for row in rows)}
    \\hline
\\end{{tabular}}
\\end{{center}}"""

        return table

    def save_accuracy_table(self, output_file: str, metrics: List[str] = None):
        """Save accuracy table to a .tex file."""
        table = self.generate_accuracy_table(metrics)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(table)

        return output_path

    def save_fairness_table(
        self, output_file: str, attribute: str, metric_type: str = "difference", fairness_metrics: List[str] = None
    ):
        """Save fairness table to a .tex file."""
        table = self.generate_fairness_table(attribute, metric_type, fairness_metrics)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(table)

        return output_path

    def save_accuracy_by_group_table(self, output_file: str, attribute: str, metrics: List[str] = None):
        """Save accuracy by group table to a .tex file."""
        table = self.generate_accuracy_by_group_table(attribute, metrics)
        if table is None:
            return None

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(table)

        return output_path

    def save_accuracy_by_group_csv(self, output_file: str, attribute: str, metrics: List[str] = None):
        """Save accuracy by group CSV to a file."""
        csv_content = self.generate_accuracy_by_group_csv(attribute, metrics)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(csv_content)

        return output_path

    def generate_fairness_table(
        self, attribute: str, metric_type: str = "difference", fairness_metrics: List[str] = None
    ) -> str:
        """
        Generate LaTeX table for fairness metrics of a specific attribute.

        Args:
            attribute: Sensitive attribute name (e.g., "gender", "age")
            metric_type: "difference" or "ratio"
            fairness_metrics: List of metrics. Default: ["demographic_parity", "equalized_odds"]

        Returns:
            LaTeX table string
        """
        if not self.fairness_results:
            raise ValueError("No fairness results available")

        if fairness_metrics is None:
            fairness_metrics = ["demographic_parity", "equalized_odds"]

        # Build header
        metric_headers = []
        for metric in fairness_metrics:
            if metric == "demographic_parity":
                metric_headers.append("Demographic Parity")
            elif metric == "equalized_odds":
                metric_headers.append("Equalized Odds")
            else:
                metric_headers.append(metric.replace("_", " ").title())

        header = "Modelo & " + " & ".join(metric_headers) + " \\\\"

        # Build rows
        rows = []
        for model_name, model_results in self.fairness_results.items():
            model_display = self.MODEL_NAME_MAP.get(model_name, model_name.title())

            # Get fairness metrics for this attribute
            user_fairness = model_results.get("aggregated", {}).get("user_fairness", {})

            if attribute not in user_fairness:
                continue

            attr_data = user_fairness[attribute]
            fairness_data = attr_data.get("fairness_metrics", {})

            # Extract values
            values = []
            for metric in fairness_metrics:
                if metric in fairness_data:
                    metric_data = fairness_data[metric]
                    if isinstance(metric_data, dict):
                        value = metric_data.get(metric_type)
                        if value is not None:
                            # Handle aggregated results (mean/std)
                            if isinstance(value, dict) and "mean" in value:
                                values.append(f"{value['mean']:.4f}")
                            elif not isinstance(value, dict):
                                values.append(f"{value:.4f}")
                            else:
                                values.append("--")
                        else:
                            values.append("--")
                    else:
                        values.append("--")
                else:
                    values.append("--")

            row = f"{model_display} & " + " & ".join(values) + " \\\\"
            rows.append(row)

        if not rows:
            raise ValueError(f"No fairness data found for attribute: {attribute}")

        # Build complete table
        col_format = "|l|" + "c " * len(fairness_metrics) + "|"

        table = f"""\\begin{{center}}
\\begin{{tabular}}{{ {col_format} }}
    \\hline
    {header}
    \\hline
    {chr(10).join(f'    {row}' for row in rows)}
    \\hline
\\end{{tabular}}
\\end{{center}}"""

        return table

    def generate_accuracy_by_group_table(self, attribute: str, metrics: List[str] = None) -> str:
        """
        Generate LaTeX table for accuracy metrics by group.

        Args:
            attribute: Sensitive attribute name
            metrics: List of metrics. Default: ["rmse", "mae"]

        Returns:
            LaTeX table string or None if too many groups (>5)
        """
        if not self.fairness_results:
            raise ValueError("No fairness results available")

        if metrics is None:
            metrics = ["rmse", "mae"]

        # Get groups from first model
        first_model = list(self.fairness_results.keys())[0]
        user_fairness = self.fairness_results[first_model].get("aggregated", {}).get("user_fairness", {})

        if attribute not in user_fairness:
            raise ValueError(f"Attribute {attribute} not found in fairness results")

        groups_data = user_fairness[attribute].get("groups", {})
        group_names = list(groups_data.keys())

        # Custom ordering for age groups
        if attribute == "age" and set(group_names) == {"young", "adult", "senior"}:
            group_names = ["young", "adult", "senior"]

        if len(group_names) > 5:
            return None  # Too many groups, should use CSV

        # Build header with multirow/multicolumn
        metric_headers_upper = []
        metric_headers_lower = []

        for group in group_names:
            # Capitalize first letter for display
            group_display = group.capitalize()
            metric_headers_upper.append(f"\\multicolumn{{{len(metrics)}}}{{c|}}{{{group_display}}}")
            metric_headers_lower.extend([m.upper() for m in metrics])

        header_upper = "\\multirow{2}{*}{Modelo} & " + " & ".join(metric_headers_upper) + " \\\\"
        header_lower = "\\cline{2-" + str(len(group_names) * len(metrics) + 1) + "}"
        header_lower += "\n     & " + " & ".join(metric_headers_lower) + " \\\\"

        # Build rows
        rows = []
        for model_name, model_results in self.fairness_results.items():
            model_display = self.MODEL_NAME_MAP.get(model_name, model_name.title())

            user_fairness = model_results.get("aggregated", {}).get("user_fairness", {})
            attr_data = user_fairness.get(attribute, {})
            groups = attr_data.get("groups", {})

            values = []
            for group in group_names:
                group_data = groups.get(group, {})
                accuracy_metrics = group_data.get("accuracy_metrics", {})

                for metric in metrics:
                    if metric in accuracy_metrics:
                        metric_value = accuracy_metrics[metric]
                        if isinstance(metric_value, dict) and "mean" in metric_value:
                            values.append(f"{metric_value['mean']:.4f}")
                        else:
                            values.append(f"{metric_value:.4f}")
                    else:
                        values.append("--")

            row = f"{model_display} & " + " & ".join(values) + " \\\\"
            rows.append(row)

        # Build complete table
        col_format = "|l|" + ("c " * len(metrics) + "|") * len(group_names)

        table = f"""\\begin{{center}}
\\begin{{tabular}}{{ {col_format} }}
    \\hline
    {header_upper}
    {header_lower}
    \\hline
    {chr(10).join(f'    {row}' for row in rows)}
    \\hline
\\end{{tabular}}
\\end{{center}}"""

        return table

    def generate_accuracy_by_group_csv(self, attribute: str, metrics: List[str] = None) -> str:
        """
        Generate CSV for accuracy metrics by group (for attributes with >5 groups).

        Args:
            attribute: Sensitive attribute name
            metrics: List of metrics. Default: ["rmse", "mae"]

        Returns:
            CSV string
        """
        if not self.fairness_results:
            raise ValueError("No fairness results available")

        if metrics is None:
            metrics = ["rmse", "mae"]

        # Get groups from first model
        first_model = list(self.fairness_results.keys())[0]
        user_fairness = self.fairness_results[first_model].get("aggregated", {}).get("user_fairness", {})

        if attribute not in user_fairness:
            raise ValueError(f"Attribute {attribute} not found in fairness results")

        groups_data = user_fairness[attribute].get("groups", {})
        group_names = sorted(groups_data.keys())

        # Build header
        header_parts = ["Model"]
        for group in group_names:
            for metric in metrics:
                header_parts.append(f"{group}_{metric.upper()}")

        csv_lines = [",".join(header_parts)]

        # Build rows
        for model_name, model_results in self.fairness_results.items():
            model_display = self.MODEL_NAME_MAP.get(model_name, model_name.title())

            user_fairness = model_results.get("aggregated", {}).get("user_fairness", {})
            attr_data = user_fairness.get(attribute, {})
            groups = attr_data.get("groups", {})

            row_parts = [model_display]
            for group in group_names:
                group_data = groups.get(group, {})
                accuracy_metrics = group_data.get("accuracy_metrics", {})

                for metric in metrics:
                    if metric in accuracy_metrics:
                        metric_value = accuracy_metrics[metric]
                        if isinstance(metric_value, dict) and "mean" in metric_value:
                            row_parts.append(f"{metric_value['mean']:.4f}")
                        else:
                            row_parts.append(f"{metric_value:.4f}")
                    else:
                        row_parts.append("")

            csv_lines.append(",".join(row_parts))

        return "\n".join(csv_lines)

    def get_user_attributes(self) -> List[str]:
        """Get list of available user fairness attributes."""
        if not self.fairness_results:
            return []

        first_model = list(self.fairness_results.keys())[0]
        user_fairness = self.fairness_results[first_model].get("aggregated", {}).get("user_fairness", {})

        return list(user_fairness.keys())

    def get_item_attributes(self) -> List[str]:
        """Get list of available item fairness attributes with valid data."""
        if not self.fairness_results:
            return []

        first_model = list(self.fairness_results.keys())[0]
        item_fairness = self.fairness_results[first_model].get("aggregated", {}).get("item_fairness", {})

        # Filter out attributes with no valid data
        valid_attributes = []
        for attr, attr_data in item_fairness.items():
            groups = attr_data.get("groups", {})
            fairness_metrics = attr_data.get("fairness_metrics", {})

            # Check if has groups or valid metrics
            has_groups = len(groups) > 0
            has_valid_metrics = any(
                v.get("ratio") is not None or v.get("difference") is not None
                for v in fairness_metrics.values()
                if isinstance(v, dict)
            )

            if has_groups or has_valid_metrics:
                valid_attributes.append(attr)

        return valid_attributes

    def generate_item_fairness_table(
        self, attribute: str, metric_type: str = "difference", fairness_metrics: List[str] = None
    ) -> str:
        """
        Generate LaTeX table for item fairness metrics of a specific attribute.

        Args:
            attribute: Sensitive attribute name (e.g., "primary_genre", "category")
            metric_type: "difference" or "ratio"
            fairness_metrics: List of metrics. Default: ["demographic_parity", "equalized_odds"]

        Returns:
            LaTeX table string
        """
        if not self.fairness_results:
            raise ValueError("No fairness results available")

        if fairness_metrics is None:
            fairness_metrics = ["demographic_parity", "equalized_odds"]

        # Build header
        metric_headers = []
        for metric in fairness_metrics:
            if metric == "demographic_parity":
                metric_headers.append("Demographic Parity")
            elif metric == "equalized_odds":
                metric_headers.append("Equalized Odds")
            else:
                metric_headers.append(metric.replace("_", " ").title())

        header = "Modelo & " + " & ".join(metric_headers) + " \\\\"

        # Build rows
        rows = []
        for model_name, model_results in self.fairness_results.items():
            model_display = self.MODEL_NAME_MAP.get(model_name, model_name.title())

            # Get fairness metrics for this attribute
            item_fairness = model_results.get("aggregated", {}).get("item_fairness", {})

            if attribute not in item_fairness:
                continue

            attr_data = item_fairness[attribute]
            fairness_data = attr_data.get("fairness_metrics", {})

            # Extract values
            values = []
            for metric in fairness_metrics:
                if metric in fairness_data:
                    metric_data = fairness_data[metric]
                    if isinstance(metric_data, dict):
                        value = metric_data.get(metric_type)
                        if value is not None:
                            # Handle aggregated results (mean/std)
                            if isinstance(value, dict) and "mean" in value:
                                values.append(f"{value['mean']:.4f}")
                            elif not isinstance(value, dict):
                                values.append(f"{value:.4f}")
                            else:
                                values.append("--")
                        else:
                            values.append("--")
                    else:
                        values.append("--")
                else:
                    values.append("--")

            row = f"{model_display} & " + " & ".join(values) + " \\\\"
            rows.append(row)

        if not rows:
            raise ValueError(f"No item fairness data found for attribute: {attribute}")

        # Build complete table
        col_format = "|l|" + "c " * len(fairness_metrics) + "|"

        table = f"""\\begin{{center}}
\\begin{{tabular}}{{ {col_format} }}
    \\hline
    {header}
    \\hline
    {chr(10).join(f'    {row}' for row in rows)}
    \\hline
\\end{{tabular}}
\\end{{center}}"""

        return table

    def generate_item_accuracy_by_group_table(self, attribute: str, metrics: List[str] = None) -> str:
        """
        Generate LaTeX table for item accuracy metrics by group.

        Args:
            attribute: Sensitive attribute name
            metrics: List of metrics. Default: ["rmse", "mae"]

        Returns:
            LaTeX table string or None if too many groups (>5)
        """
        if not self.fairness_results:
            raise ValueError("No fairness results available")

        if metrics is None:
            metrics = ["rmse", "mae"]

        # Get groups from first model
        first_model = list(self.fairness_results.keys())[0]
        item_fairness = self.fairness_results[first_model].get("aggregated", {}).get("item_fairness", {})

        if attribute not in item_fairness:
            raise ValueError(f"Attribute {attribute} not found in item fairness results")

        groups_data = item_fairness[attribute].get("groups", {})
        group_names = list(groups_data.keys())

        # Custom ordering for age groups
        if attribute == "age" and set(group_names) == {"young", "adult", "senior"}:
            group_names = ["young", "adult", "senior"]

        if len(group_names) > 5:
            return None  # Too many groups, should use CSV

        # Build header with multirow/multicolumn
        metric_headers_upper = []
        metric_headers_lower = []

        for group in group_names:
            # Capitalize first letter for display
            group_display = group.capitalize()
            metric_headers_upper.append(f"\\multicolumn{{{len(metrics)}}}{{c|}}{{{group_display}}}")
            metric_headers_lower.extend([m.upper() for m in metrics])

        header_upper = "\\multirow{2}{*}{Modelo} & " + " & ".join(metric_headers_upper) + " \\\\"
        header_lower = "\\cline{2-" + str(len(group_names) * len(metrics) + 1) + "}"
        header_lower += "\n     & " + " & ".join(metric_headers_lower) + " \\\\"

        # Build rows
        rows = []
        for model_name, model_results in self.fairness_results.items():
            model_display = self.MODEL_NAME_MAP.get(model_name, model_name.title())

            item_fairness = model_results.get("aggregated", {}).get("item_fairness", {})
            attr_data = item_fairness.get(attribute, {})
            groups = attr_data.get("groups", {})

            values = []
            for group in group_names:
                group_data = groups.get(group, {})
                accuracy_metrics = group_data.get("accuracy_metrics", {})

                for metric in metrics:
                    if metric in accuracy_metrics:
                        metric_value = accuracy_metrics[metric]
                        if isinstance(metric_value, dict) and "mean" in metric_value:
                            values.append(f"{metric_value['mean']:.4f}")
                        else:
                            values.append(f"{metric_value:.4f}")
                    else:
                        values.append("--")

            row = f"{model_display} & " + " & ".join(values) + " \\\\"
            rows.append(row)

        # Build complete table
        col_format = "|l|" + ("c " * len(metrics) + "|") * len(group_names)

        table = f"""\\begin{{center}}
\\begin{{tabular}}{{ {col_format} }}
    \\hline
    {header_upper}
    {header_lower}
    \\hline
    {chr(10).join(f'    {row}' for row in rows)}
    \\hline
\\end{{tabular}}
\\end{{center}}"""

        return table

    def generate_item_accuracy_by_group_csv(self, attribute: str, metrics: List[str] = None) -> str:
        """
        Generate CSV for item accuracy metrics by group (for attributes with >5 groups).

        Args:
            attribute: Sensitive attribute name
            metrics: List of metrics. Default: ["rmse", "mae"]

        Returns:
            CSV string
        """
        if not self.fairness_results:
            raise ValueError("No fairness results available")

        if metrics is None:
            metrics = ["rmse", "mae"]

        # Get groups from first model
        first_model = list(self.fairness_results.keys())[0]
        item_fairness = self.fairness_results[first_model].get("aggregated", {}).get("item_fairness", {})

        if attribute not in item_fairness:
            raise ValueError(f"Attribute {attribute} not found in item fairness results")

        groups_data = item_fairness[attribute].get("groups", {})
        group_names = sorted(groups_data.keys())

        # Build header
        header_parts = ["Model"]
        for group in group_names:
            for metric in metrics:
                header_parts.append(f"{group}_{metric.upper()}")

        csv_lines = [",".join(header_parts)]

        # Build rows
        for model_name, model_results in self.fairness_results.items():
            model_display = self.MODEL_NAME_MAP.get(model_name, model_name.title())

            item_fairness = model_results.get("aggregated", {}).get("item_fairness", {})
            attr_data = item_fairness.get(attribute, {})
            groups = attr_data.get("groups", {})

            row_parts = [model_display]
            for group in group_names:
                group_data = groups.get(group, {})
                accuracy_metrics = group_data.get("accuracy_metrics", {})

                for metric in metrics:
                    if metric in accuracy_metrics:
                        metric_value = accuracy_metrics[metric]
                        if isinstance(metric_value, dict) and "mean" in metric_value:
                            row_parts.append(f"{metric_value['mean']:.4f}")
                        else:
                            row_parts.append(f"{metric_value:.4f}")
                    else:
                        row_parts.append("")

            csv_lines.append(",".join(row_parts))

        return "\n".join(csv_lines)

    def save_item_fairness_table(
        self, output_file: str, attribute: str, metric_type: str = "difference", fairness_metrics: List[str] = None
    ):
        """Save item fairness table to a .tex file."""
        table = self.generate_item_fairness_table(attribute, metric_type, fairness_metrics)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(table)

        return output_path

    def save_item_accuracy_by_group_table(self, output_file: str, attribute: str, metrics: List[str] = None):
        """Save item accuracy by group table to a .tex file."""
        table = self.generate_item_accuracy_by_group_table(attribute, metrics)
        if table is None:
            return None

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(table)

        return output_path

    def save_item_accuracy_by_group_csv(self, output_file: str, attribute: str, metrics: List[str] = None):
        """Save item accuracy by group CSV to a file."""
        csv_content = self.generate_item_accuracy_by_group_csv(attribute, metrics)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(csv_content)

        return output_path
