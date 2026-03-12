# HH.ru Salary Prediction Pipeline

Пайплайн обработки резюме с hh.ru и регрессионная модель для предсказания зарплаты.
Архитектура основана на паттерне "Цепочка ответственности" (Chain of Responsibility).

## Структура проекта

```
.
├── app.py                          # Предсказание зарплаты по .npy файлу признаков
├── train.py                        # Обучение и сохранение модели
├── run_pipeline.py                 # Запуск пайплайна обработки CSV → .npy
├── requirements.txt
├── resources/
│   └── salary_model.pkl            # Веса модели (генерируется train.py, не в git)
├── data/
│   └── hh.csv                      # Исходные данные (не в git)
└── src/
    ├── pipeline_builder.py         # Построитель пайплайна
    ├── core/
    │   ├── handler.py              # Абстрактный обработчик
    │   └── pipeline_context.py     # Контекст пайплайна
    ├── handlers/
    │   ├── data_loader.py          # Загрузка CSV
    │   ├── data_cleaner.py         # Очистка данных
    │   ├── advanced_feature_extractor.py  # Извлечение признаков из резюме
    │   ├── feature_engineering.py  # Кодирование категориальных признаков
    │   ├── data_splitter.py        # Разделение на X и y
    │   ├── data_normalizer.py      # Нормализация
    │   └── data_exporter.py        # Сохранение в .npy
    └── model/
        ├── constants.py            # Константы модели
        ├── model_trainer.py        # Обучение GradientBoostingRegressor
        └── salary_predictor.py     # Загрузка модели и инференс
```

## Требования

- Python 3.8+
- RAM: ~2 GB (обучение на ~47k сэмплах)
- CPU: обучение занимает ~10 секунд на современном процессоре

## Установка

```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Использование

### 1. Подготовка данных (CSV → .npy)

```bash
python run_pipeline.py data/hh.csv
```

Создаёт `data/x_data.npy` и `data/y_data.npy`.

### 2. Обучение модели

```bash
python train.py data/x_data.npy data/y_data.npy
```

Сохраняет обученную модель в `resources/salary_model.pkl`.
Выводит метрики на тестовой выборке (MAE и R²).

### 3. Предсказание зарплаты

```bash
python app.py data/x_data.npy
```

Выводит список предсказанных зарплат в рублях:

```
[40755.62, 69533.0, 73597.62, ...]
```

Логи уходят в stderr и не мешают stdout-выводу.

## Паттерн Chain of Responsibility

Каждый обработчик выполняет одну задачу и передаёт контекст дальше:

1. **DataLoaderHandler** — загружает CSV
2. **AdvancedFeatureExtractorHandler** — парсит поля резюме (возраст, опыт, образование и др.)
3. **DataCleanerHandler** — удаляет дубликаты и пропуски
4. **DataSplitterHandler** — разделяет на признаки X и таргет y (зарплата)
5. **DataNormalizerHandler** — стандартизация признаков
6. **DataExporterHandler** — сохраняет результат в .npy

## Модель

`GradientBoostingRegressor` (scikit-learn) с log-преобразованием таргета:

- Выбросы по зарплате фильтруются (< 5 000 или > 500 000 руб)
- Таргет логарифмируется (`log1p`) перед обучением, предсказания переводятся обратно (`expm1`)
- Метрики на тестовой выборке: **MAE ≈ 30 500 руб**, **R² ≈ 0.39**
