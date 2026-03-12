"""Builds a numeric feature matrix from raw IT resume data."""

import logging
import re
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.classification.constants import (
    COL_CAR,
    COL_CITY,
    COL_EDUCATION,
    COL_EMPLOYMENT,
    COL_EXPERIENCE,
    COL_GENDER_AGE,
    COL_SALARY,
    COL_SCHEDULE,
    OTHER_CITY_LABEL,
    TOP_CITIES_COUNT,
)

logger = logging.getLogger(__name__)

_YEARS_PATTERN = re.compile(r"(\d+)\s+(?:год|года|лет)")
_MONTHS_PATTERN = re.compile(r"(\d+)\s+месяц")
_AGE_PATTERN = re.compile(r"(\d+)\s+(?:год|года|лет)")
_SALARY_DIGITS = re.compile(r"\d+")


class FeatureBuilder:
    """Transforms raw resume columns into a numeric feature matrix.

    Must be fitted on training data before transforming.  Fitting records
    the top cities list so that unseen cities are mapped to 'Other'.

    Attributes:
        top_cities: List of the most frequent city names seen during fit.
    """

    def __init__(self) -> None:
        """Initialize the feature builder with an empty city list."""
        self.top_cities: List[str] = []

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """Fit on df and return the feature matrix together with column names.

        Args:
            df: IT-filtered dataframe with original raw columns.

        Returns:
            Tuple of (feature matrix, list of feature names).
        """
        self.top_cities = (
            df[COL_CITY]
            .apply(self._parse_city)
            .value_counts()
            .head(TOP_CITIES_COUNT)
            .index.tolist()
        )
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Transform df using the city list recorded during fit.

        Args:
            df: Dataframe with the same raw columns used during fit.

        Returns:
            Tuple of (feature matrix, list of feature names).
        """
        parts: List[pd.DataFrame] = [
            self._base_features(df),
            self._employment_features(df),
            self._schedule_features(df),
            self._city_features(df),
        ]
        result = pd.concat(parts, axis=1)
        logger.info("Built feature matrix: %s", result.shape)
        return result.values.astype(np.float32), list(result.columns)

    # ------------------------------------------------------------------
    # Private helpers — each extracts one group of features
    # ------------------------------------------------------------------

    def _base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract scalar numeric features.

        Args:
            df: Raw dataframe.

        Returns:
            DataFrame with columns: gender, age, salary,
            experience_years, experience_months, education_level, has_car.
        """
        gender_age = df[COL_GENDER_AGE].apply(self._parse_gender_age)
        experience = df[COL_EXPERIENCE].apply(self._parse_experience)
        return pd.DataFrame(
            {
                "gender": [x[0] for x in gender_age],
                "age": [x[1] for x in gender_age],
                "salary": df[COL_SALARY].apply(self._parse_salary),
                "experience_years": [x[0] for x in experience],
                "experience_months": [x[1] for x in experience],
                "education_level": df[COL_EDUCATION].apply(self._parse_education),
                "has_car": df[COL_CAR].apply(self._parse_car),
            }
        )

    def _employment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract binary employment-type flags.

        Args:
            df: Raw dataframe.

        Returns:
            DataFrame with one binary column per employment type.
        """
        records = df[COL_EMPLOYMENT].apply(self._parse_employment)
        return pd.DataFrame(list(records))

    def _schedule_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract binary work-schedule flags.

        Args:
            df: Raw dataframe.

        Returns:
            DataFrame with one binary column per schedule type.
        """
        records = df[COL_SCHEDULE].apply(self._parse_schedule)
        return pd.DataFrame(list(records))

    def _city_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode city, grouping infrequent cities as 'Other'.

        Args:
            df: Raw dataframe.

        Returns:
            One-hot encoded DataFrame with city indicator columns.
        """
        cities = df[COL_CITY].apply(self._parse_city)
        top = set(self.top_cities)
        mapped = cities.apply(lambda c, t=top: c if c in t else OTHER_CITY_LABEL)
        return pd.get_dummies(mapped, prefix="city")

    # ------------------------------------------------------------------
    # Atomic parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_gender_age(value: str) -> Tuple[int, int]:
        """Parse gender (0/1) and age from 'Пол, возраст' field.

        Args:
            value: Raw field string.

        Returns:
            Tuple of (gender, age): gender 1 = male, 0 = female; age in years.
        """
        gender = 1 if "Мужчина" in str(value) else 0
        match = _AGE_PATTERN.search(str(value))
        age = int(match.group(1)) if match else 0
        return gender, age

    @staticmethod
    def _parse_salary(value: str) -> int:
        """Extract numeric salary from ЗП field, converting KZT to RUB.

        Args:
            value: Raw salary string.

        Returns:
            Salary in rubles as integer, or 0 if unparseable.
        """
        digits = _SALARY_DIGITS.findall(str(value))
        if not digits:
            return 0
        salary = int("".join(digits))
        if "KZT" in str(value).upper():
            salary = int(salary * 0.2)
        return salary

    @staticmethod
    def _parse_experience(value: str) -> Tuple[int, int]:
        """Extract years and months from the experience field.

        Args:
            value: Raw experience field text.

        Returns:
            Tuple of (years, months).
        """
        years_match = _YEARS_PATTERN.search(str(value))
        months_match = _MONTHS_PATTERN.search(str(value))
        years = int(years_match.group(1)) if years_match else 0
        months = int(months_match.group(1)) if months_match else 0
        return years, months

    @staticmethod
    def _parse_education(value: str) -> int:
        """Map education description to an ordinal level.

        Args:
            value: Raw education field text.

        Returns:
            3 = higher, 2 = vocational secondary, 1 = secondary, 0 = unknown.
        """
        text = str(value).lower()
        if "высшее" in text:
            return 3
        if "среднее специальное" in text or "техникум" in text:
            return 2
        if "среднее" in text:
            return 1
        return 0

    @staticmethod
    def _parse_car(value: str) -> int:
        """Detect car ownership from the 'Авто' field.

        Args:
            value: Raw car field text.

        Returns:
            1 if the candidate owns a car, 0 otherwise.
        """
        return 1 if "собственный автомобиль" in str(value).lower() else 0

    @staticmethod
    def _parse_city(value: str) -> str:
        """Extract bare city name from the 'Город' field.

        Args:
            value: Raw city field string, e.g. 'Москва , не готов к переезду'.

        Returns:
            City name only.
        """
        text = str(value)
        return text.split(",")[0].strip() if "," in text else text.strip()

    @staticmethod
    def _parse_employment(value: str) -> dict:
        """Produce binary flags for each employment type.

        Args:
            value: Raw employment field string.

        Returns:
            Dictionary mapping employment flag names to 0 or 1.
        """
        text = str(value).lower()
        return {
            "emp_full_time": int("полная занятость" in text or "full time" in text),
            "emp_part_time": int("частичная занятость" in text or "part time" in text),
            "emp_project": int("проектная работа" in text or "project work" in text),
            "emp_internship": int("стажировка" in text or "internship" in text),
            "emp_volunteer": int("волонтерство" in text or "volunteering" in text),
        }

    @staticmethod
    def _parse_schedule(value: str) -> dict:
        """Produce binary flags for each work schedule type.

        Args:
            value: Raw schedule field string.

        Returns:
            Dictionary mapping schedule flag names to 0 or 1.
        """
        text = str(value).lower()
        return {
            "sched_full_day": int("полный день" in text or "full day" in text),
            "sched_flexible": int("гибкий график" in text or "flexible" in text),
            "sched_shift": int("сменный график" in text or "shift" in text),
            "sched_remote": int("удаленная работа" in text or "remote" in text),
            "sched_rotation": int("вахтовый метод" in text or "rotation" in text),
        }
