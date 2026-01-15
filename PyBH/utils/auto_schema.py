from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
from utils.config import CATEGORICAL_THRESHOLD_COUNT


@dataclass
class DatasetSchema:
    duration_col: str = None
    start_col: str = None
    event_col: str = None
    numerical_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    dropped_cols: List[str] = field(default_factory=list)


class SchemaDetector:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.schema = DatasetSchema()

        # 1. PRIORITY LIST: CENSORSHIP
        # If we find a 'censor' column, we prefer it (to invert it later)
        self.censor_keywords = ["cens", "censor", "censored"]

        # 2. SECONDARY LIST: CLASSIC EVENT
        self.event_keywords = [
            "event",
            "status",
            "dead",
            "death",
            "fstat",
            "statut",
            "deces",
            "outcome",
            "died",
            "recurrence",
            "relapse",
            "fail",
        ]

        # 3. TIME Keywords
        self.all_time_keywords = [
            "time",
            "duration",
            "days",
            "months",
            "years",
            "survival",
            "duree",
            "temps",
            "suivi",
            "futime",
            "ttdeath",
            "start",
            "begin",
            "debut",
            "entry",
            "enter",
            "t0",
            "stop",
            "end",
            "fin",
            "exit",
            "t1",
            "date",
        ]

        # Start/End Sub-lists
        self.start_keywords = [
            "start",
            "begin",
            "debut",
            "time0",
            "t0",
            "entry",
            "enter",
        ]
        self.end_keywords = [
            "stop",
            "end",
            "fin",
            "exit",
            "time1",
            "t1",
            "duration",
            "duree",
            "survival",
        ]

        self.id_keywords = ["id", "patient", "ref", "index", "subject", "matricule"]

    def infer(self) -> DatasetSchema:
        cols = list(self.df.columns)

        # --- STEP 1: Find Event (Priority Logic) ---

        # A. First explicitly look for a Censorship column
        censor_col = self._find_best_match(
            cols, self.censor_keywords, dtype_filter="number"
        )

        if censor_col:
            # If 'cens' found, take it
            self.schema.event_col = censor_col
        else:
            # Otherwise, look for a classic 'event' column
            self.schema.event_col = self._find_best_match(
                cols, self.event_keywords, dtype_filter="number"
            )

        # --- STEP 2: Scan TIME candidates ---
        search_space = [c for c in cols if c != self.schema.event_col]
        time_candidates = []

        for col in search_space:
            if self._is_id_column(col):
                continue

            if self._matches_keyword(col, self.all_time_keywords):
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    time_candidates.append(col)

        # --- STEP 3: Time Decision Logic ---
        if len(time_candidates) > 2:
            raise ValueError(
                f"Time ambiguity ({len(time_candidates)} candidate columns). "
                "Please configure config.py."
            )
        elif len(time_candidates) == 1:
            self.schema.duration_col = time_candidates[0]
        elif len(time_candidates) == 2:
            col_a, col_b = time_candidates[0], time_candidates[1]
            score_a = self._get_role_score(col_a)
            score_b = self._get_role_score(col_b)
            if score_a > score_b:
                self.schema.start_col, self.schema.duration_col = col_a, col_b
            else:
                self.schema.start_col, self.schema.duration_col = col_b, col_a

        # --- STEP 4: Classify the Rest ---
        exclude_cols = [
            self.schema.duration_col,
            self.schema.start_col,
            self.schema.event_col,
        ]
        remaining_cols = [c for c in cols if c not in exclude_cols and c is not None]

        for col in remaining_cols:
            if self._is_id_column(col):
                self.schema.dropped_cols.append(col)
                continue

            if self._is_categorical(col):
                self.schema.categorical_cols.append(col)
            else:
                self.schema.numerical_cols.append(col)

        return self.schema

    def _matches_keyword(self, col_name, keywords):
        col_lower = col_name.lower()
        if col_lower in keywords:
            return True
        for kw in keywords:
            if kw in col_lower:
                return True
        return False

    def _find_best_match(self, columns, keywords, dtype_filter=None) -> Optional[str]:
        for kw in keywords:
            for col in columns:
                if col.lower() == kw:
                    return col
        for kw in keywords:
            for col in columns:
                if kw in col.lower():
                    if dtype_filter == "number" and not pd.api.types.is_numeric_dtype(
                        self.df[col]
                    ):
                        continue
                    return col
        return None

    def _get_role_score(self, col_name):
        col_lower = col_name.lower()
        score = 0
        if any(kw in col_lower for kw in self.start_keywords):
            score += 10
        if any(kw in col_lower for kw in self.end_keywords):
            score -= 10
        if "0" in col_lower:
            score += 5
        if "1" in col_lower:
            score -= 5
        return score

    def _is_id_column(self, col: str) -> bool:
        if any(x in col.lower() for x in self.id_keywords):
            return True
        if self.df[col].dtype == object and self.df[col].nunique() == len(self.df):
            return True
        return False

    def _is_categorical(self, col: str) -> bool:
        if self.df[col].dtype == object or self.df[col].dtype.name == "category":
            return True
        if pd.api.types.is_float_dtype(self.df[col]):
            return False
        if pd.api.types.is_integer_dtype(self.df[col]):
            if self.df[col].nunique() <= CATEGORICAL_THRESHOLD_COUNT:
                return True
        return False
