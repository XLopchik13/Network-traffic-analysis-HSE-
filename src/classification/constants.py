"""Constants for IT developer classification."""

# --- Class labels ---
LEVEL_JUNIOR = "junior"
LEVEL_MIDDLE = "middle"
LEVEL_SENIOR = "senior"
LEVELS = [LEVEL_JUNIOR, LEVEL_MIDDLE, LEVEL_SENIOR]

# --- Experience thresholds (years) ---
EXP_JUNIOR_THRESHOLD = 2
EXP_SENIOR_THRESHOLD = 5

# --- Keywords for IT developer detection ---
IT_KEYWORDS = [
    "разработчик",
    "программист",
    "developer",
    "software engineer",
    "python",
    "java",
    "javascript",
    "php",
    "c++",
    "c#",
    ".net",
    "backend",
    "frontend",
    "fullstack",
    "full-stack",
    "devops",
    "android",
    "ios",
    "mobile",
    "1с",
    "1c",
    "data scientist",
    "machine learning",
    "веб-разработчик",
    "инженер-программист",
]

# --- Keywords for level detection in job titles ---
SENIOR_KEYWORDS = [
    "senior",
    "сеньор",
    "старший",
    "ведущий",
    "lead",
    "главный",
    "principal",
    "architect",
    "архитектор",
    "тимлид",
    "team lead",
    "tech lead",
]

JUNIOR_KEYWORDS = [
    "junior",
    "джуниор",
    "младший",
    "начинающий",
    "стажёр",
    "стажер",
    "intern",
]

# --- Raw CSV column names ---
COL_TITLE = "Ищет работу на должность:"
COL_LAST_POSITION = "Последеняя/нынешняя должность"
COL_EXPERIENCE = "Опыт (двойное нажатие для полной версии)"
COL_GENDER_AGE = "Пол, возраст"
COL_SALARY = "ЗП"
COL_CITY = "Город"
COL_EMPLOYMENT = "Занятость"
COL_SCHEDULE = "График"
COL_EDUCATION = "Образование и ВУЗ"
COL_CAR = "Авто"

# --- Feature engineering ---
TOP_CITIES_COUNT = 20
OTHER_CITY_LABEL = "Other"

# --- Output ---
PLOTS_DIR = "resources/plots"
CLASS_BALANCE_PLOT = "class_balance.png"
FEATURE_IMPORTANCE_PLOT = "feature_importance.png"
TOP_FEATURES_COUNT = 15
