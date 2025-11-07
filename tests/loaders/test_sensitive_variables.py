import pandas as pd
import pytest

from .conftest import assert_mappings_exist, assert_sensitive_variables_structure


class TestSensitiveVariables:
    def test_users_sensitive_variables_structure(self, safe_loader, dataset_name, dataset_sensitive_config):
        users_config = dataset_sensitive_config.get("users", {})
        if not users_config:
            pytest.skip(f"No users config defined for {dataset_name}")

        users = safe_loader.get_users()
        assert_sensitive_variables_structure(users, users_config, "users")

    def test_items_sensitive_variables_structure(self, safe_loader, dataset_name, dataset_sensitive_config):
        items_config = dataset_sensitive_config.get("items", {})
        if not items_config:
            pytest.skip(f"No items config defined for {dataset_name}")

        items = safe_loader.get_items()
        assert_sensitive_variables_structure(items, items_config, "items")

    def test_users_mappings_exist(self, safe_loader, dataset_name, dataset_sensitive_config):
        users_config = dataset_sensitive_config.get("users", {})
        if not users_config.get("mappings"):
            pytest.skip(f"No user mappings expected for {dataset_name}")

        safe_loader.get_users()
        assert_mappings_exist(safe_loader, users_config, "users")

    def test_items_mappings_exist(self, safe_loader, dataset_name, dataset_sensitive_config):
        items_config = dataset_sensitive_config.get("items", {})
        if not items_config.get("mappings"):
            pytest.skip(f"No item mappings expected for {dataset_name}")

        safe_loader.get_items()
        assert_mappings_exist(safe_loader, items_config, "items")

    def test_mappings_consistency(self, safe_loader, dataset_name, dataset_sensitive_config):
        users_config = dataset_sensitive_config.get("users", {})
        if not users_config.get("mappings"):
            pytest.skip(f"No user mappings to test for {dataset_name}")

        users = safe_loader.get_users()
        user_mappings = safe_loader.get_user_mappings()

        for col in users_config.get("mappings", []):
            if col in users.columns:
                unique_values = users[col].unique()
                mapping_values = set(user_mappings[col].values())
                for val in unique_values:
                    if not pd.isna(val):
                        assert val in mapping_values, f"Value {val} not found in {col} mapping for {dataset_name}"

    def test_decode_invalid_variable(self, safe_loader):
        safe_loader.get_users()

        result = safe_loader.decode_user_variable("invalid_var", 0)
        assert result == 0

        user_mappings = safe_loader.get_user_mappings()
        if user_mappings:
            first_var = next(iter(user_mappings.keys()))
            result = safe_loader.decode_user_variable(first_var, 999)
            assert result == 999

    @pytest.mark.parametrize("entity_type", ["users", "items"])
    def test_basic_dataframe_requirements(self, safe_loader, entity_type):
        if entity_type == "users":
            df = safe_loader.get_users()
            required_col = "user_id"
        else:
            df = safe_loader.get_items()
            required_col = "item_id"

        assert isinstance(df, pd.DataFrame)
        assert required_col in df.columns
        if len(df) > 0:
            assert df[required_col].notna().all()
