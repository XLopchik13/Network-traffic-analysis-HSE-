# HH.ru Resume Analysis

Три независимых ML-модуля поверх одного датасета резюме с hh.ru:

1. **Регрессия (GBR)** — предсказание зарплаты (`GradientBoostingRegressor`)
2. **Регрессия (FCN)** — нейросеть для предсказания зарплаты с трекингом в MLflow
3. **Классификация** — уровень IT-разработчика: junior / middle / senior (`RandomForestClassifier`)

Пайплайн обработки данных построен на паттерне "Цепочка ответственности" (Chain of Responsibility).

---

## Структура проекта

```
.
├── app.py                              # Предсказание зарплаты по .npy признакам
├── train.py                            # Обучение регрессионной модели
├── run_pipeline.py                     # Пайплайн: CSV → x_data.npy + y_data.npy
├── run_classification_poc.py           # PoC классификации уровня разработчика
├── requirements.txt
│
├── resources/
│   ├── salary_model.pkl                # Веса GBR (генерируется, не в git)
│   ├── fcn_model.pt                    # Веса FCN (генерируется, не в git)
│   └── plots/                          # Графики классификации
│       ├── class_balance.png
│       └── feature_importance.png
│
├── data/
│   └── hh.csv                          # Исходные данные (не в git)
│
└── src/
    ├── pipeline_builder.py
    ├── core/
    │   ├── handler.py                  # Абстрактный обработчик (Chain of Responsibility)
    │   └── pipeline_context.py         # Контекст, передаваемый по цепочке
    ├── handlers/
    │   ├── data_loader.py
    │   ├── data_cleaner.py
    │   ├── advanced_feature_extractor.py
    │   ├── feature_engineering.py
    │   ├── data_splitter.py
    │   ├── data_normalizer.py
    │   └── data_exporter.py
    ├── model/                          # Регрессия зарплаты
    │   ├── constants.py                # Константы GBR, FCN и MLflow
    │   ├── model_trainer.py            # Обучение GradientBoostingRegressor
    │   ├── salary_predictor.py         # Инференс GBR
    │   ├── fcn_model.py                # PyTorch FCN архитектура
    │   └── neural_trainer.py           # Обучение FCN + ранняя остановка
    └── classification/                 # Классификация уровня разработчика
        ├── constants.py
        ├── it_filter.py
        ├── level_labeler.py
        ├── feature_builder.py
        └── developer_classifier.py
```

---

## Требования

- Python 3.8+
- RAM: ~2 GB (обучение на ~47k сэмплах)
- CPU: GBR — ~10 с, FCN — ~30–60 с (CPU, 45 эпох с early stopping)

## Установка

```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

## Модуль 1 — Предсказание зарплаты

### 1. Подготовка данных (CSV → .npy)

```bash
python run_pipeline.py data/hh.csv
```

Создаёт `data/x_data.npy` (признаки) и `data/y_data.npy` (зарплата).

### 2. Обучение модели

```bash
python train.py data/x_data.npy data/y_data.npy
```

Сохраняет модель в `resources/salary_model.pkl`, выводит MAE и R².

### 3. Предсказание

```bash
python app.py data/x_data.npy
```

Вывод в stdout — список зарплат в рублях:

```
[40755.62, 69533.0, 73597.62, ...]
```

Логи идут в stderr и не мешают stdout-выводу.

### Модели и метрики

| Параметр | GBR | FCN (нейросеть) |
|---|---|---|
| Алгоритм | `GradientBoostingRegressor` | PyTorch, 4 слоя [512→256→128→64] |
| Таргет | `log1p(salary)` | `log1p(salary)` |
| Фильтрация выбросов | < 5 000 и > 500 000 руб | < 5 000 и > 500 000 руб |
| MAE (тест) | ≈ 30 500 руб | ≈ 32 100 руб |
| R² (тест) | ≈ 0.39 | ≈ 0.38 |
| Трекинг | — | MLflow |

---

## Модуль 1б — FCN нейросеть + MLflow

### Обучение с трекингом

```bash
python train_neural.py data/x_data.npy data/y_data.npy
```

Скрипт:
- подключается к MLflow-серверу `http://kamnsv.com:55000/`
- создаёт run в эксперименте **"LIne Regression HH"**
- логирует гиперпараметры, `r2_score_test`, `mae_test`
- регистрирует модель как **`petrov_nikita_nikolaevich_fcn`**
- сохраняет веса локально в `resources/fcn_model.pt`
- выводит `RUN_ID` в stdout

### Архитектура FCN

```
Input (36) → Linear(512) → BN → ReLU
           → Linear(256) → BN → ReLU
           → Linear(128) → BN → ReLU
           → Linear(64)  → BN → ReLU
           → Linear(1)
```

- Оптимизатор: Adam (lr=5e-4, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau (patience=15, factor=0.5)
- Early stopping: patience=30 эпох
- Dropout убран — с BN избыточен и вызывал underfitting

### MLflow параметры

| Параметр | Значение |
|---|---|
| Tracking URI | `http://kamnsv.com:55000/` |
| Experiment | `LIne Regression HH` |
| Model name | `petrov_nikita_nikolaevich_fcn` |
| Metric | `r2_score_test` |
| Best RUN_ID | `f5de9c8ec4384a0d809d7c60cde36087` |

---

## Модуль 2 — Классификация уровня IT-разработчика (PoC)

```bash
python run_classification_poc.py data/hh.csv
```

**Что делает:**

1. Фильтрует IT-резюме по ключевым словам в желаемой/текущей должности
2. Размечает уровень (приоритет: ключевые слова в названии → стаж работы)
3. Строит и сохраняет график баланса классов
4. Обучает `RandomForestClassifier` с `class_weight='balanced'`
5. Выводит classification report (precision / recall / F1 по классам)
6. Сохраняет график важности признаков
7. Печатает выводы о качестве и ограничениях модели

**Результаты на датасете:**

| Уровень | Резюме | Precision | Recall | F1 |
|---|---|---|---|---|
| junior | 3 018 (22%) | 0.93 | 0.81 | 0.86 |
| middle | 1 930 (14%) | 0.85 | 0.96 | 0.90 |
| senior | 8 578 (63%) | 0.95 | 0.97 | 0.96 |
| **macro avg** | 13 526 | **0.91** | **0.91** | **0.91** |

Графики сохраняются в `resources/plots/`.

---

## Паттерн Chain of Responsibility

Пайплайн построен как цепочка независимых обработчиков — каждый решает одну задачу:

1. **DataLoaderHandler** — загрузка CSV
2. **AdvancedFeatureExtractorHandler** — парсинг полей резюме (возраст, опыт, образование и др.)
3. **DataCleanerHandler** — удаление дубликатов и пропусков
4. **DataSplitterHandler** — разделение на X и y
5. **DataNormalizerHandler** — стандартизация
6. **DataExporterHandler** — сохранение в .npy

---

## Запуск тестов

```bash
python -c "from src.model.salary_predictor import SalaryPredictor; print('OK')"
python -c "from src.model.fcn_model import FCNModel; print('OK')"
python -c "from src.classification.developer_classifier import DeveloperClassifier; print('OK')"
```
