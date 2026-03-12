"""Filter for extracting IT developer rows from hh.ru resume data."""

import logging

import pandas as pd

from src.classification.constants import COL_LAST_POSITION, COL_TITLE, IT_KEYWORDS

logger = logging.getLogger(__name__)


class ITFilter:
    """Selects rows that correspond to IT developer positions.

    A row is considered IT-related when either the desired job title or
    the last held position contains at least one IT keyword.
    """

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only rows where the candidate targets or holds an IT role.

        Args:
            df: Raw resume dataframe with all original columns.

        Returns:
            Filtered dataframe reset to a new integer index.
        """
        title_mask = df[COL_TITLE].apply(self._is_it_title)
        last_pos_mask = df[COL_LAST_POSITION].apply(self._is_it_title)
        it_df = df[title_mask | last_pos_mask].reset_index(drop=True)
        logger.info(
            "IT filter: %d / %d rows kept (%.1f%%)",
            len(it_df),
            len(df),
            100 * len(it_df) / len(df),
        )
        return it_df

    def _is_it_title(self, title: str) -> bool:
        """Check whether a job title contains an IT keyword.

        Args:
            title: Raw job title string from the resume.

        Returns:
            True if the title matches at least one IT keyword.
        """
        text = str(title).lower()
        return any(keyword in text for keyword in IT_KEYWORDS)
