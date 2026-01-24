# Network Traffic Analysis Pipeline

Пайплайн обработки данных сетевого трафика с использованием паттерна проектирования "Цепочка ответственности" (Chain of Responsibility).

## Описание

Проект реализует чистую архитектуру для обработки данных с полной типизацией, логированием и модульной структурой. Пайплайн преобразует CSV файл с данными в NumPy массивы, готовые для машинного обучения.

## Структура проекта

```
.
├── app.py                      # Точка входа приложения
├── src/
│   ├── __init__.py
│   ├── pipeline_builder.py    # Построитель пайплайна
│   ├── core/                   # Базовые классы архитектуры
│   │   ├── __init__.py
│   │   ├── handler.py          # Абстрактный обработчик
│   │   └── pipeline_context.py # Контекст пайплайна
│   └── handlers/               # Конкретные обработчики
│       ├── __init__.py
│       ├── data_loader.py      # Загрузка данных
│       ├── data_cleaner.py     # Очистка данных
│       ├── feature_engineering.py  # Инженерия признаков
│       ├── data_splitter.py    # Разделение на X и y
│       ├── data_normalizer.py  # Нормализация данных
│       └── data_exporter.py    # Экспорт в .npy
└── data/
    └── hh.csv                  # Исходные данные
```

## Паттерн Chain of Responsibility

Каждый обработчик в цепочке выполняет одну задачу и передает контекст следующему:

1. **DataLoaderHandler** - загружает CSV файл
2. **DataCleanerHandler** - удаляет дубликаты и пропуски
3. **FeatureEngineeringHandler** - кодирует категориальные признаки
4. **DataSplitterHandler** - разделяет на признаки (X) и целевую переменную (y)
5. **DataNormalizerHandler** - нормализует признаки (стандартизация)
6. **DataExporterHandler** - сохраняет результат в .npy файлы

## Требования

- Python 3.8+
- pandas
- numpy

## Установка

```bash
# Создать виртуальное окружение
python -m venv venv

# Активировать виртуальное окружение (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Установить зависимости
pip install -r requirements.txt
```

## Использование

```bash
# Запуск пайплайна
python app.py data/hh.csv
```

Команда создаст два файла в директории с исходным CSV:
- `x_data.npy` - матрица признаков
- `y_data.npy` - вектор целевой переменной
