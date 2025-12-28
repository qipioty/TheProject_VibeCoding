```markdown
# Study Material Recommender

[![Tests](https://github.com/qipioty/study-materials-recommender/actions/workflows/tests.yml/badge.svg)](https://github.com/qipioty/study-materials-recommender/actions/workflows/tests.yml)
[![Retrain Workflow](https://github.com/qipioty/study-materials-recommender/actions/workflows/schedule.yml/badge.svg)](https://github.com/qipioty/study-materials-recommender/actions/workflows/schedule.yml)
[![Pages](https://github.com/qipioty/study-materials-recommender/workflows/Deploy%20to%20Pages/badge.svg)](https://github.com/qipioty/study-materials-recommender/actions/workflows/deploy-gh-pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Краткое описание
----------------
Простой гибридный рекомендатель учебных материалов для студентов.
Комбинирует:
- content-based (TF-IDF на title + description)
- latent item factors (TruncatedSVD на item-user матрице)

Проект демонстрирует рабочую рекомендательную систему с тестами и CI/CD,
включая scheduled retrain и deploy отчёта на GitHub Pages.

Быстрый старт
------------
1. Клонируй репозиторий и создай виртуальное окружение:
```bash
git clone https://github.com/qipioty/study-materials-recommender.git
cd study-materials-recommender
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Запусти демо:
```bash
python src/main.py
```

3. Запусти тесты и линтер:
```bash
pytest tests/ -v
flake8 src tests
black --check src tests || true
```

Про CI/CD
---------
- .github/workflows/tests.yml — lint, black-check, pytest + coverage, upload artifacts, отправка в codecov (опционально).
- .github/workflows/schedule.yml — еженедельный retrain (cron) + workflow_dispatch; генерирует retrain_report.csv и docs/report.html и загружает model.joblib и CSV как артефакты.
- .github/workflows/deploy-gh-pages.yml — публикует содержимое директории docs/ в ветку gh-pages (GitHub Pages).

Файлы и структура
-----------------
.
├── src/                 # Исходники (реализация рекоммендера, утилиты)
├── tests/               # Unit tests (pytest)
├── data/                # Примеры данных (sample CSV)
├── scripts/             # Утилиты: retrain script
├── docs/                # Генерируемые отчёты (deploy -> gh-pages)
├── .github/workflows/   # CI workflows
├── README.md
├── requirements.txt
├── .flake8
└── .gitignore

Как сдавать
----------
1. Убедись, что весь код закоммичен в ветку `main`.
2. Проверь Actions (Tests и Retrain). Убедись, что последний run — success.
3. Отправь ссылку на репозиторий преподавателю.

Идеи для улучшений
------------------
- Добавить реальные данные LMS и механизмы автоматического обновления dataset.
- Улучшить модель (matrix factorization, implicit feedback, LightFM, ALS).
- Добавить web API (FastAPI) и CI для деплоя сервиса.

Лицензия
--------
MIT
```