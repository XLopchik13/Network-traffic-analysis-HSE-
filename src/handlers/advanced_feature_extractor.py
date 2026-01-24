"""Advanced feature extraction handler for hh.ru resumes."""

from typing import Optional
import pandas as pd
import numpy as np
import logging
import re
from datetime import datetime

from ..core.handler import Handler
from ..core.pipeline_context import PipelineContext


class AdvancedFeatureExtractorHandler(Handler):
    """Handler for extracting structured features from raw resume data.
    
    This handler parses text fields and extracts meaningful numeric features:
    - Gender and age from 'Пол, возраст'
    - Salary from 'ЗП'
    - Experience years and months from 'Опыт'
    - Education level from 'Образование и ВУЗ'
    - And more...
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Initialize the feature extractor handler.
        
        Args:
            logger: Logger instance. If None, creates a new logger.
        """
        super().__init__(logger)
        
    def _extract_gender_age(self, row: str) -> tuple[int, int]:
        """Extract gender and age from 'Пол, возраст' field.
        
        Args:
            row: String like 'Мужчина , 42 года , родился 6 октября 1976'
            
        Returns:
            Tuple of (gender, age) where gender: 1=male, 0=female
        """
        gender = 1 if 'Мужчина' in str(row) else 0
        age_match = re.search(r'(\d+)\s+(?:год|года|лет)', str(row))
        age = int(age_match.group(1)) if age_match else 0
        return gender, age
        
    def _extract_salary(self, row: str) -> int:
        """Extract numeric salary from 'ЗП' field.
        
        Args:
            row: String like '27 000 руб.' or '60 000 руб.' or '100 000 KZT'
            
        Returns:
            Salary as integer (в рублях, KZT конвертируется).
        """
        row_str = str(row)
        digits = re.findall(r'\d+', row_str)
        if not digits:
            return 0

        salary_str = ''.join(digits)
        try:
            salary = int(salary_str)
            if 'KZT' in row_str.upper():
                salary = int(salary * 0.2)
            return salary
        except:
            return 0
        
    def _extract_experience(self, row: str) -> tuple[int, int]:
        """Extract years and months of experience.
        
        Args:
            row: String containing 'Опыт работы X лет Y месяцев'
            
        Returns:
            Tuple of (years, months).
        """
        years_match = re.search(r'(\d+)\s+(?:год|года|лет)', str(row))
        months_match = re.search(r'(\d+)\s+месяц', str(row))
        
        years = int(years_match.group(1)) if years_match else 0
        months = int(months_match.group(1)) if months_match else 0
        
        return years, months
        
    def _extract_has_car(self, row: str) -> int:
        """Check if person has a car.
        
        Args:
            row: String from 'Авто' field.
            
        Returns:
            1 if has car, 0 otherwise.
        """
        return 1 if 'собственный автомобиль' in str(row).lower() else 0
        
    def _extract_education_level(self, row: str) -> int:
        """Extract education level.
        
        Args:
            row: String from 'Образование и ВУЗ' field.
            
        Returns:
            Education level: 3=Высшее, 2=Среднее специальное, 1=Среднее, 0=Не указано.
        """
        text = str(row).lower()
        if 'высшее' in text:
            return 3
        elif 'среднее специальное' in text or 'техникум' in text:
            return 2
        elif 'среднее' in text:
            return 1
        return 0
        
    def _extract_city(self, row: str) -> str:
        """Extract city name from 'Город' field.
        
        Args:
            row: String like 'Москва , не готов к переезду , не готов к командировкам'
            
        Returns:
            City name only.
        """
        text = str(row)
        if ',' in text:
            return text.split(',')[0].strip()
        return text.strip()
        
    def _parse_employment(self, row: str) -> dict[str, int]:
        """Parse employment types as multi-label.
        
        Args:
            row: String like 'полная занятость, частичная занятость' or 'full time, part time'
            
        Returns:
            Dict with binary flags for each employment type.
        """
        text = str(row).lower()
        return {
            'employment_full_time': 1 if ('полная занятость' in text or 'full time' in text) else 0,
            'employment_part_time': 1 if ('частичная занятость' in text or 'part time' in text) else 0,
            'employment_project': 1 if ('проектная работа' in text or 'project work' in text) else 0,
            'employment_internship': 1 if ('стажировка' in text or 'work placement' in text or 'internship' in text) else 0,
            'employment_volunteer': 1 if ('волонтерство' in text or 'volunteering' in text) else 0,
        }
        
    def _parse_schedule(self, row: str) -> dict[str, int]:
        """Parse work schedule as multi-label.
        
        Args:
            row: String like 'полный день, удаленная работа' or 'full day, remote working'
            
        Returns:
            Dict with binary flags for each schedule type.
        """
        text = str(row).lower()
        return {
            'schedule_full_day': 1 if ('полный день' in text or 'full day' in text) else 0,
            'schedule_flexible': 1 if ('гибкий график' in text or 'flexible schedule' in text) else 0,
            'schedule_shift': 1 if ('сменный график' in text or 'shift schedule' in text) else 0,
            'schedule_remote': 1 if ('удаленная работа' in text or 'remote working' in text or 'remote work' in text) else 0,
            'schedule_rotation': 1 if ('вахтовый метод' in text or 'rotation based work' in text) else 0,
        }
        
    def process(self, context: PipelineContext) -> PipelineContext:
        """Extract features from raw data.
        
        Args:
            context: The pipeline context with data to process.
            
        Returns:
            Context with extracted features.
            
        Raises:
            ValueError: If data is None.
        """
        if context.data is None:
            error_msg = "No data for feature extraction"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        data = context.data.copy()
        self.logger.info(f"Extracting features from {len(data)} rows")

        if 'Unnamed: 0' in data.columns:
            self.logger.info("Dropping index column (Unnamed: 0)")
            data = data.drop('Unnamed: 0', axis=1)

        if 'Пол, возраст' in data.columns:
            self.logger.info("Extracting gender and age...")
            gender_age = data['Пол, возраст'].apply(self._extract_gender_age)
            data['gender'] = [x[0] for x in gender_age]
            data['age'] = [x[1] for x in gender_age]
            data = data.drop('Пол, возраст', axis=1)

        if 'ЗП' in data.columns:
            self.logger.info("Extracting salary...")
            data['salary'] = data['ЗП'].apply(self._extract_salary)
            data = data.drop('ЗП', axis=1)

        if 'Опыт (двойное нажатие для полной версии)' in data.columns:
            self.logger.info("Extracting experience...")
            experience = data['Опыт (двойное нажатие для полной версии)'].apply(
                self._extract_experience
            )
            data['experience_years'] = [x[0] for x in experience]
            data['experience_months'] = [x[1] for x in experience]
            data = data.drop('Опыт (двойное нажатие для полной версии)', axis=1)

        if 'Авто' in data.columns:
            self.logger.info("Extracting car ownership...")
            data['has_car'] = data['Авто'].apply(self._extract_has_car)
            data = data.drop('Авто', axis=1)

        if 'Образование и ВУЗ' in data.columns:
            self.logger.info("Extracting education level...")
            data['education_level'] = data['Образование и ВУЗ'].apply(
                self._extract_education_level
            )
            data = data.drop('Образование и ВУЗ', axis=1)

        if 'Занятость' in data.columns:
            self.logger.info("Parsing employment as multi-label...")
            employment = data['Занятость'].apply(self._parse_employment)
            for key in ['employment_full_time', 'employment_part_time', 'employment_project', 
                       'employment_internship', 'employment_volunteer']:
                data[key] = [e[key] for e in employment]
            data = data.drop('Занятость', axis=1)

        if 'График' in data.columns:
            self.logger.info("Parsing schedule as multi-label...")
            schedule = data['График'].apply(self._parse_schedule)
            for key in ['schedule_full_day', 'schedule_flexible', 'schedule_shift', 
                       'schedule_remote', 'schedule_rotation']:
                data[key] = [s[key] for s in schedule]
            data = data.drop('График', axis=1)

        if 'Город' in data.columns:
            self.logger.info("Extracting city names...")
            data['city'] = data['Город'].apply(self._extract_city)
            data = data.drop('Город', axis=1)

        text_cols_to_drop = [
            'Ищет работу на должность:',
            'Последенее/нынешнее место работы',
            'Последеняя/нынешняя должность',
            'Обновление резюме'
        ]
        for col in text_cols_to_drop:
            if col in data.columns:
                self.logger.info(f"Dropping text column: {col}")
                data = data.drop(col, axis=1)

        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            self.logger.info(f"One-hot encoding {len(categorical_cols)} categorical columns...")
            for col in categorical_cols:
                top_categories = data[col].value_counts().head(20).index
                data[col] = data[col].apply(lambda x: x if x in top_categories else 'Other')
            data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
            
        self.logger.info(f"Feature extraction complete. New shape: {data.shape}")
        context.update_data(data)
        context.add_metadata("extracted_features", True)
        context.add_metadata("feature_count", data.shape[1])
        
        return context
