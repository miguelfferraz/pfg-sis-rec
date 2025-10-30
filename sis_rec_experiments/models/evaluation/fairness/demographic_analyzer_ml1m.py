from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class DemographicAnalyzerML1M:
    """
    Analisador demográfico específico para MovieLens 1M.
    Adapta os grupos etários conforme solicitado: Adult + Middle_Age = Adult.
    """
    
    def __init__(self):
        # Mapeamento dos códigos de idade do ML-1M para grupos
        # Baseado no README: 1="Under 18", 18="18-24", 25="25-34", 35="35-44", 45="45-49", 50="50-55", 56="56+"
        self.age_code_to_group = {
            1: "Young",      # Under 18
            18: "Young",     # 18-24  
            25: "Adult",     # 25-34
            35: "Adult",     # 35-44
            45: "Adult",     # 45-49 (era Middle_Age, agora Adult)
            50: "Adult",     # 50-55 (era Middle_Age, agora Adult)
            56: "Senior"     # 56+
        }
        
        # Mapeamento das ocupações para categorias
        self.occupation_categories = {
            "Professional": [1, 6, 7, 11, 12, 15, 17],  # academic, doctor, executive, lawyer, programmer, scientist, technician
            "Education": [1, 4, 10],                     # academic, college student, K-12 student
            "Creative": [2, 20],                         # artist, writer
            "Service": [3, 5, 14],                       # clerical, customer service, sales
            "Technical": [12, 17],                       # programmer, technician
            "Other": [0, 8, 9, 13, 16, 18, 19]         # other, farmer, homemaker, retired, self-employed, tradesman, unemployed
        }

    def categorize_age_ml1m(self, age_code: int) -> str:
        """Categoriza código de idade do ML-1M em grupos etários"""
        return self.age_code_to_group.get(age_code, "Other")

    def categorize_occupation_ml1m(self, occupation_code: int) -> str:
        """Categoriza código de ocupação do ML-1M em categorias"""
        for category, codes in self.occupation_categories.items():
            if occupation_code in codes:
                return category
        return "Other"

    def process_demographics_ml1m(self, user_demographics: pd.DataFrame) -> pd.DataFrame:
        """
        Processa dados demográficos do MovieLens 1M.
        
        Args:
            user_demographics: DataFrame com colunas [user_id, gender, age, occupation, zip_code]
        
        Returns:
            DataFrame processado com grupos categorizados
        """
        df = user_demographics.copy()
        
        # Categoriza idade (códigos → grupos)
        df["age_group"] = df["age"].apply(self.categorize_age_ml1m)
        
        # Categoriza ocupação (códigos → categorias)
        df["occupation_category"] = df["occupation"].apply(self.categorize_occupation_ml1m)
        
        # Normaliza gênero
        df["gender"] = df["gender"].map({"M": "Male", "F": "Female"})
        
        return df

    def get_group_statistics(self, demographics: pd.DataFrame) -> Dict[str, any]:
        """Calcula estatísticas dos grupos demográficos"""
        
        stats = {}
        
        # Estatísticas por feature individual
        for feature in ["gender", "age_group", "occupation_category"]:
            if feature in demographics.columns:
                counts = demographics[feature].value_counts().to_dict()
                stats[feature] = counts
        
        # Cross-tabulations (convertendo tuplas para strings para JSON)
        if "gender" in demographics.columns and "age_group" in demographics.columns:
            cross_tab = demographics.groupby(["gender", "age_group"]).size().to_dict()
            stats["gender_age_cross"] = {f"{k[0]}_{k[1]}": v for k, v in cross_tab.items()}
        
        if "gender" in demographics.columns and "occupation_category" in demographics.columns:
            cross_tab = demographics.groupby(["gender", "occupation_category"]).size().to_dict()
            stats["gender_occupation_cross"] = {f"{k[0]}_{k[1]}": v for k, v in cross_tab.items()}
        
        return stats

    def identify_minority_groups(self, demographics: pd.DataFrame, threshold: float = 0.05) -> Dict[str, List[str]]:
        """Identifica grupos minoritários (< threshold da população)"""
        
        minority_groups = {}
        total_users = len(demographics)
        
        for feature in ["gender", "age_group", "occupation_category"]:
            if feature in demographics.columns:
                counts = demographics[feature].value_counts()
                percentages = counts / total_users
                
                minorities = percentages[percentages < threshold].index.tolist()
                if minorities:
                    minority_groups[feature] = minorities
        
        return minority_groups

    def create_sensitive_features_matrix(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """Cria matriz de features sensíveis para análise de fairness"""
        
        features_df = demographics[["user_id"]].copy()
        
        # Adiciona features sensíveis disponíveis
        for feature in ["gender", "age_group", "occupation_category"]:
            if feature in demographics.columns:
                features_df[feature] = demographics[feature]
        
        return features_df

    def get_age_mapping_info(self) -> Dict[str, any]:
        """Retorna informações sobre o mapeamento de idades"""
        
        return {
            "age_code_mapping": self.age_code_to_group,
            "age_groups_description": {
                "Young": "Under 18 + 18-24 years (codes 1, 18)",
                "Adult": "25-34 + 35-44 + 45-49 + 50-55 years (codes 25, 35, 45, 50)",
                "Senior": "56+ years (code 56)"
            },
            "note": "Adult group combines original Adult and Middle_Age as requested"
        }

    def analyze_ml1m_distribution(self, user_demographics: pd.DataFrame) -> Dict[str, any]:
        """Analisa a distribuição específica do MovieLens 1M"""
        
        processed_df = self.process_demographics_ml1m(user_demographics)
        
        analysis = {
            "total_users": len(processed_df),
            "original_age_codes": user_demographics["age"].value_counts().sort_index().to_dict(),
            "grouped_ages": processed_df["age_group"].value_counts().to_dict(),
            "gender_distribution": processed_df["gender"].value_counts().to_dict(),
            "occupation_categories": processed_df["occupation_category"].value_counts().to_dict(),
            "age_mapping_used": self.get_age_mapping_info()
        }
        
        return analysis

