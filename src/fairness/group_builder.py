from typing import Any, Dict, List

import pandas as pd


class GroupBuilder:
    def __init__(self, attribute_config: Dict[str, Any], loader, entity_type: str = "user"):
        self.attribute_config = attribute_config
        self.loader = loader
        self.entity_type = entity_type
        self.attribute_name = attribute_config["name"]
        self.attribute_type = attribute_config["type"]

    def build_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_groups = df.copy()
        group_column_name = f"group_{self.attribute_name}"

        if self.attribute_type == "categorical":
            df_with_groups[group_column_name] = self._build_categorical_groups(df)
        elif self.attribute_type == "numeric":
            df_with_groups[group_column_name] = self._build_numeric_groups(df)
        else:
            raise ValueError(f"Unknown attribute type: {self.attribute_type}")

        return df_with_groups

    def _build_categorical_groups(self, df: pd.DataFrame) -> pd.Series:
        if self.attribute_name not in df.columns:
            return pd.Series([None] * len(df), index=df.index)

        encoded_values = df[self.attribute_name]
        labels = []

        decode_method = (
            self.loader.decode_user_variable if self.entity_type == "user" else self.loader.decode_item_variable
        )

        for encoded_value in encoded_values:
            if pd.isna(encoded_value):
                labels.append(None)
            else:
                label = decode_method(self.attribute_name, int(encoded_value))
                labels.append(str(label))

        return pd.Series(labels, index=df.index)

    def _build_numeric_groups(self, df: pd.DataFrame) -> pd.Series:
        if self.attribute_name not in df.columns:
            return pd.Series([None] * len(df), index=df.index)

        bins_config = self.attribute_config.get("bins", [])
        if not bins_config:
            raise ValueError(f"Numeric attribute '{self.attribute_name}' requires 'bins' configuration")

        values = df[self.attribute_name]
        labels = []

        for value in values:
            if pd.isna(value):
                labels.append(None)
                continue

            assigned = False
            for bin_config in bins_config:
                if bin_config["min"] <= value <= bin_config["max"]:
                    labels.append(bin_config["label"])
                    assigned = True
                    break

            if not assigned:
                labels.append(None)

        return pd.Series(labels, index=df.index)

    def get_group_labels(self, df: pd.DataFrame) -> List[str]:
        group_column_name = f"group_{self.attribute_name}"
        if group_column_name not in df.columns:
            df = self.build_groups(df)

        unique_groups = df[group_column_name].dropna().unique().tolist()
        return sorted(unique_groups)
