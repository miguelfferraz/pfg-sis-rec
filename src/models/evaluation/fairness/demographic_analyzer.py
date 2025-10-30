from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class DemographicAnalyzer:
    def __init__(self):
        self.age_groups = {"Young": (0, 25), "Adult": (26, 45), "Middle_Age": (46, 60), "Senior": (61, 100)}

        self.occupation_categories = {
            "Professional": ["doctor", "lawyer", "engineer", "scientist", "programmer", "executive"],
            "Education": ["educator", "student", "librarian"],
            "Creative": ["artist", "writer", "entertainment"],
            "Service": ["healthcare", "administrator", "marketing", "salesman"],
            "Technical": ["technician"],
            "Other": ["homemaker", "retired", "none", "other"],
        }

    def categorize_age(self, age: int) -> str:
        for group, (min_age, max_age) in self.age_groups.items():
            if min_age <= age <= max_age:
                return group
        return "Other"

    def categorize_occupation(self, occupation: str) -> str:
        occupation_lower = occupation.lower()

        for category, occupations in self.occupation_categories.items():
            if occupation_lower in occupations:
                return category

        return "Other"

    def process_demographics(self, user_demographics: pd.DataFrame) -> pd.DataFrame:
        df = user_demographics.copy()

        df["age_group"] = df["age"].apply(self.categorize_age)
        df["occupation_category"] = df["occupation"].apply(self.categorize_occupation)
        df["gender"] = df["gender"].map({"M": "Male", "F": "Female"})

        return df

    def get_group_statistics(self, demographics: pd.DataFrame) -> Dict:
        stats = {}

        stats["gender"] = demographics["gender"].value_counts().to_dict()
        stats["age_group"] = demographics["age_group"].value_counts().to_dict()
        stats["occupation_category"] = demographics["occupation_category"].value_counts().to_dict()

        # Converte tuplas para strings para serialização JSON
        gender_age_cross = demographics.groupby(["gender", "age_group"]).size().to_dict()
        stats["gender_age_cross"] = {f"{k[0]}_{k[1]}": v for k, v in gender_age_cross.items()}
        
        gender_occupation_cross = demographics.groupby(["gender", "occupation_category"]).size().to_dict()
        stats["gender_occupation_cross"] = {f"{k[0]}_{k[1]}": v for k, v in gender_occupation_cross.items()}

        return stats

    def identify_minority_groups(self, demographics: pd.DataFrame, threshold: float = 0.1) -> Dict[str, List[str]]:
        total_users = len(demographics)
        minority_groups = {}

        for feature in ["gender", "age_group", "occupation_category"]:
            if feature in demographics.columns:
                counts = demographics[feature].value_counts()
                proportions = counts / total_users

                minority_groups[feature] = [group for group, prop in proportions.items() if prop < threshold]

        return minority_groups

    def create_sensitive_features_matrix(self, demographics: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        available_features = [f for f in features if f in demographics.columns]

        if not available_features:
            raise ValueError(f"Nenhuma das features {features} encontrada nos dados demográficos")

        return demographics[["user_id"] + available_features].copy()
