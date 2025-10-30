"""
Testes para módulos de fairness.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sis_rec_experiments.models.evaluation.fairness import (
    DemographicAnalyzer,
    FairnessEvaluator,
    FairnessMetricsCalculator,
)


class TestDemographicAnalyzer(unittest.TestCase):
    """Testes para DemographicAnalyzer"""

    def setUp(self):
        self.analyzer = DemographicAnalyzer()

        # Dados de teste
        self.test_demographics = pd.DataFrame(
            {
                "user_id": [1, 2, 3, 4, 5, 6],
                "age": [25, 45, 65, 30, 20, 50],
                "gender": ["M", "F", "M", "F", "M", "F"],
                "occupation": ["student", "engineer", "retired", "doctor", "artist", "educator"],
                "zip_code": ["12345", "67890", "11111", "22222", "33333", "44444"],
            }
        )

    def test_categorize_age(self):
        """Testa categorização de idade"""
        self.assertEqual(self.analyzer.categorize_age(20), "Young")
        self.assertEqual(self.analyzer.categorize_age(35), "Adult")
        self.assertEqual(self.analyzer.categorize_age(55), "Middle_Age")
        self.assertEqual(self.analyzer.categorize_age(70), "Senior")

    def test_categorize_occupation(self):
        """Testa categorização de ocupação"""
        self.assertEqual(self.analyzer.categorize_occupation("engineer"), "Professional")
        self.assertEqual(self.analyzer.categorize_occupation("student"), "Education")
        self.assertEqual(self.analyzer.categorize_occupation("artist"), "Creative")
        self.assertEqual(self.analyzer.categorize_occupation("retired"), "Other")

    def test_process_demographics(self):
        """Testa processamento de dados demográficos"""
        processed = self.analyzer.process_demographics(self.test_demographics)

        # Verifica se as colunas foram adicionadas
        self.assertIn("age_group", processed.columns)
        self.assertIn("occupation_category", processed.columns)
        self.assertIn("gender", processed.columns)

        # Verifica alguns valores
        self.assertEqual(processed.loc[0, "age_group"], "Young")  # age 25
        self.assertEqual(processed.loc[1, "occupation_category"], "Professional")  # engineer
        self.assertEqual(processed.loc[0, "gender"], "Male")  # M -> Male

    def test_get_group_statistics(self):
        """Testa cálculo de estatísticas por grupo"""
        processed = self.analyzer.process_demographics(self.test_demographics)
        stats = self.analyzer.get_group_statistics(processed)

        # Verifica estrutura
        self.assertIn("gender", stats)
        self.assertIn("age_group", stats)
        self.assertIn("occupation_category", stats)

        # Verifica valores
        self.assertEqual(stats["gender"]["Male"], 3)
        self.assertEqual(stats["gender"]["Female"], 3)


class TestFairnessMetricsCalculator(unittest.TestCase):
    """Testes para FairnessMetricsCalculator"""

    def setUp(self):
        self.calculator = FairnessMetricsCalculator(rating_threshold=4.0)

        # Dados de teste
        self.y_true = np.array([5, 3, 4, 2, 5, 1])
        self.y_pred = np.array([4.5, 3.2, 4.1, 2.8, 4.8, 1.5])
        self.sensitive_features = pd.Series(["Male", "Female", "Male", "Female", "Male", "Female"])

    def test_convert_to_binary(self):
        """Testa conversão para classificação binária"""
        binary = self.calculator._convert_to_binary(self.y_pred)
        expected = np.array([1, 0, 1, 0, 1, 0])  # >= 4.0
        np.testing.assert_array_equal(binary, expected)

    def test_calculate_demographic_parity(self):
        """Testa cálculo de paridade demográfica"""
        result = self.calculator.calculate_demographic_parity(self.y_true, self.y_pred, self.sensitive_features)

        # Verifica estrutura
        self.assertIn("demographic_parity_difference", result)
        self.assertIn("demographic_parity_ratio", result)
        self.assertIn("selection_rate_by_group", result)

        # Verifica tipos
        self.assertIsInstance(result["demographic_parity_difference"], (int, float))
        self.assertIsInstance(result["selection_rate_by_group"], dict)

    def test_calculate_calibration_fairness(self):
        """Testa cálculo de calibração de fairness"""
        result = self.calculator.calculate_calibration_fairness(self.y_true, self.y_pred, self.sensitive_features)

        # Verifica estrutura
        self.assertIn("rmse_Male", result)
        self.assertIn("rmse_Female", result)
        self.assertIn("mae_Male", result)
        self.assertIn("mae_Female", result)
        self.assertIn("max_rmse_difference", result)
        self.assertIn("max_mae_difference", result)


class TestFairnessEvaluator(unittest.TestCase):
    """Testes para FairnessEvaluator"""

    def setUp(self):
        self.evaluator = FairnessEvaluator(sensitive_features=["gender", "age_group"], rating_threshold=4.0)

        # Mock de dados demográficos
        self.demographics = pd.DataFrame(
            {
                "user_id": [1, 2, 3, 4],
                "age": [25, 45, 30, 60],
                "gender": ["M", "F", "M", "F"],
                "occupation": ["student", "engineer", "doctor", "retired"],
                "zip_code": ["12345", "67890", "11111", "22222"],
            }
        )

    def test_extract_prediction_data(self):
        """Testa extração de dados de predição"""

        # Mock de predições da Surprise
        class MockPrediction:
            def __init__(self, uid, iid, r_ui, est):
                self.uid = uid
                self.iid = iid
                self.r_ui = r_ui
                self.est = est

        predictions = [
            MockPrediction("1", "101", 5.0, 4.5),
            MockPrediction("2", "102", 3.0, 3.2),
            MockPrediction("3", "103", 4.0, 4.1),
        ]

        df = self.evaluator._extract_prediction_data(predictions)

        # Verifica estrutura
        self.assertEqual(len(df), 3)
        self.assertIn("user_id", df.columns)
        self.assertIn("item_id", df.columns)
        self.assertIn("rating_true", df.columns)
        self.assertIn("rating_pred", df.columns)

        # Verifica valores
        self.assertEqual(df.iloc[0]["user_id"], "1")
        self.assertEqual(df.iloc[0]["rating_true"], 5.0)
        self.assertEqual(df.iloc[0]["rating_pred"], 4.5)


if __name__ == "__main__":
    unittest.main()
