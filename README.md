# Study Material Recommender

[![Tests](https://github.com/qipioty/TheProject_VibeCoding/actions/workflows/tests.yml/badge.svg)](https://github.com/qipioty/TheProject_VibeCoding/actions/workflows/tests.yml)
[![Retrain Workflow](https://github.com/qipioty/TheProject_VibeCoding/actions/workflows/schedule.yml/badge.svg)](https://github.com/qipioty/TheProject_VibeCoding/actions/workflows/schedule.yml)
[![Pages](https://github.com/qipioty/TheProject_VibeCoding/workflows/Deploy%20to%20Pages/badge.svg)](https://github.com/qipioty/TheProject_VibeCoding/actions/workflows/deploy-gh-pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Краткое описание
----------------
Проект — учебная рекомендательная система учебных материалов (Study Material Recommender).
Система комбинирует content-based (TF‑IDF по title+description) и latent-item‑factors
(TruncatedSVD на item-user матрице) подходы и демонстрирует:
- подготовку проекта по best-practices,
- тестирование и проверку качества кода (pytest, flake8, black),
- автоматическое переобучение по расписанию и загрузку артефактов,
- деплой простого отчёта на GitHub Pages.

Этот проект подготовлен как финальный проект-основа для курса (соответствует шаблону final_project_base).

Быстрый старт
------------
Требования:
- Python 3.10+ (в CI используется 3.10 / 3.11)
- git

Установка (локально)
```bash
git clone https://github.com/qipioty/TheProject_VibeCoding.git
cd TheProject_VibeCoding

# создать виртуальное окружение
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows (PowerShell)
# .\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Запуск demo
```bash
# печатает рекомендации для нескольких пользователей
python src/main.py
```

Ручное переобучение и генерация отчёта
```bash
python scripts/retrain.py
# создаст retrain_report.csv, model.joblib и docs/report.html
```

Запуск тестов и линтера (локально)
```bash
pytest tests/ -v
flake8 src tests
black --check src tests || true
```

Структура проекта
-----------------
.
├── src/                 # Исходники (реализация рекоммендера, утилиты)  
│   ├── __init__.py  
│   ├── recommender.py  
│   ├── utils.py  
│   └── main.py  
├── tests/               # Unit tests (pytest)  
│   └── test_recommender.py  
├── data/                # Примеры данных (sample CSV)  
│   ├── sample_materials.csv  
│   └── sample_ratings.csv  
├── scripts/             # Утилиты: retrain script  
├── docs/                # Генерируемые отчёты (deploy -> gh-pages)  
├── .github/workflows/   # CI workflows (tests, schedule, deploy)  
├── README.md  
├── requirements.txt  
├── .flake8  
├── .gitignore  
└── LICENSE

Что делает проект (подробно)
----------------------------
- Загружает материалы и оценки студентов (CSV).
- Строит TF‑IDF векторизацию по title + description и считает content‑similarity.
- Строит item-user матрицу, применяет TruncatedSVD для латентных признаков (с безопасным fallback для маленьких датасетов).
- Для пользователя агрегирует соседей по content и latent компонентам и выдаёт top‑k рекомендаций.
- Скрипт `scripts/retrain.py` переобучает модель и генерирует CSV/HTML отчёт (подходит для CI).

CI / GitHub Actions
-------------------
В комплекте несколько workflow:

1. `.github/workflows/tests.yml` — основной pipeline:
   - Запускается на `push`/`pull_request` в ветки `main`/`master`.
   - Устанавливает зависимости, запускает `flake8` и `black --check`.
   - Перед `pytest` в CI выставляется `PYTHONPATH=${{ github.workspace }}` (чтобы `import src` работал).
   - Запускает `pytest` с покрытием, сохраняет `pytest-report.xml` и `coverage.xml` как артефакты.
   - Загружает покрытие в codecov (опционально — при наличии токена).

2. `.github/workflows/schedule.yml` — retrain workflow:
   - Запускается по расписанию (`cron`) и вручную (`workflow_dispatch`).
   - Выполняет `scripts/retrain.py`, сохраняет `retrain_report.csv`, `docs/report.html` и `model.joblib` как артефакты.

3. `.github/workflows/deploy-gh-pages.yml` — деплой отчётов:
   - Запускается вручную или при push в `main`.
   - Генерирует/обновляет `docs/` (запускает retrain) и публикует содержимое `docs/` в ветку `gh-pages` (GitHub Pages).

Замечания по артефактам
- В workflow настроено безопасное поведение при загрузке артефактов (избежание конфликта имён): используются флаги перезаписи (`overwrite: true`) или уникальные имена артефактов.
- После успешного запуска deploy workflow отчёт будет доступен по адресу:
  https://qipioty.github.io/TheProject_VibeCoding/ (если Pages включены).

Требования / Зависимости
------------------------
Содержимое `requirements.txt`. Ключевые:
- numpy
- pandas
- scikit-learn
- joblib

Тестирование и качество кода
---------------------------
- Минимум 3–5 unit тестов (в репозитории — примеры тестов в `tests/`).
- CI запускает `pytest`, `flake8` и `black --check`.
- Локально: `pytest tests/ -v` и `flake8 src tests`.

Как сдавать проект (для курса)
-----------------------------
1. Убедитесь, что все изменения закоммичены в ветку `main`.  
2. Push в репозиторий на GitHub (адрес: https://github.com/qipioty/TheProject_VibeCoding).  
3. Убедитесь, что workflow `Tests and Code Quality` завершился успешно (Actions → Tests).  
4. (Опционально) Запустите вручную Retrain и Deploy workflow, проверьте артефакты и Pages.  
5. Отправьте ссылку на репозиторий преподавателю/в систему курса.

Идеи для улучшения (дополнительно)
---------------------------------
- Реализовать implicit-feedback подход (ALS / LightFM).  
- Добавить метрики качества (RMSE, precision@k) и автоматическую генерацию графиков.  
- Добавить web API (FastAPI) и Dockerfile + workflow для деплоя.  
- Интегрировать реальный LMS dataset и автоматическое обновление данных (CI).

Лицензия
--------
Проект распространяется под лицензией MIT — файл `LICENSE` в корне.

Контакты
--------
Автор: qipioty  
Если найдёте баги или хотите предложить улучшения — откройте issue или PR в репозитории.
```
