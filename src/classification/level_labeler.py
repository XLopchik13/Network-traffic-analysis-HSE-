"""Assigns junior / middle / senior labels to IT developer rows."""

import logging
import re
from typing import Optional

import pandas as pd

from src.classification.constants import (
    COL_EXPERIENCE,
    COL_LAST_POSITION,
    COL_TITLE,
    EXP_JUNIOR_THRESHOLD,
    EXP_SENIOR_THRESHOLD,
    JUNIOR_KEYWORDS,
    LEVEL_JUNIOR,
    LEVEL_MIDDLE,
    LEVEL_SENIOR,
    SENIOR_KEYWORDS,
)

logger = logging.getLogger(__name__)

_YEARS_PATTERN = re.compile(r"(\d+)\s+(?:год|года|лет)")


class LevelLabeler:
    """Assigns a seniority label to each IT resume row.

    Labeling priority:
    1. Keywords found in the desired job title.
    2. Keywords found in the last held position.
    3. Years of experience (fallback when no keyword matches).
    """

    def label(self, df: pd.DataFrame) -> pd.Series:
        """Produce a seniority label for every row in the dataframe.

        Args:
            df: IT-filtered resume dataframe.

        Returns:
            Series of string labels aligned with df's index.
        """
        labels = df.apply(self._label_row, axis=1)
        dist = labels.value_counts().to_dict()
        logger.info("Level distribution: %s", dist)
        return labels

    def _label_row(self, row: pd.Series) -> str:
        """Determine the seniority level for a single resume row.

        Args:
            row: One row from the IT-filtered dataframe.

        Returns:
            One of 'junior', 'middle', or 'senior'.
        """
        level = self._level_by_title(row[COL_TITLE])
        if level is not None:
            return level

        level = self._level_by_title(row[COL_LAST_POSITION])
        if level is not None:
            return level

        return self._level_by_experience(row[COL_EXPERIENCE])

    def _level_by_title(self, title: str) -> Optional[str]:
        """Detect seniority from keyword presence in a job title.

        Args:
            title: Job title string to inspect.

        Returns:
            Detected level string, or None if no keyword matches.
        """
        text = str(title).lower()
        if any(kw in text for kw in SENIOR_KEYWORDS):
            return LEVEL_SENIOR
        if any(kw in text for kw in JUNIOR_KEYWORDS):
            return LEVEL_JUNIOR
        return None

    def _level_by_experience(self, experience: str) -> str:
        """Map years of experience to a seniority level.

        Args:
            experience: Raw experience field text containing duration info.

        Returns:
            'junior' for < 2 years, 'senior' for >= 5 years, 'middle' otherwise.
        """
        years = self._parse_years(experience)
        if years < EXP_JUNIOR_THRESHOLD:
            return LEVEL_JUNIOR
        if years >= EXP_SENIOR_THRESHOLD:
            return LEVEL_SENIOR
        return LEVEL_MIDDLE

    def _parse_years(self, experience: str) -> int:
        """Extract the number of years from an experience field.

        Args:
            experience: Raw experience field text, e.g. 'Опыт работы 6 лет 1 месяц'.

        Returns:
            Number of years as an integer, or 0 if not found.
        """
        match = _YEARS_PATTERN.search(str(experience))
        return int(match.group(1)) if match else 0
