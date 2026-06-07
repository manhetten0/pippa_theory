# Pippa verification framework

Модульный Python-фреймворк для проверки базовых формул теории Pippa
и проведения симуляций.

## Установка

Требуется Python 3.10+.

### 1. Создать и активировать виртуальное окружение (venv)

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Установить зависимости

Только для запуска фреймворка (runtime):

```bash
pip install -r requirements.txt
```

Для разработки и запуска тестов:

```bash
pip install -r requirements-dev.txt
```

> Примечание: базовое ядро использует только стандартную библиотеку,
> поэтому `requirements.txt` пуст. Внешние пакеты понадобятся при
> добавлении численных симуляций.

## Запуск

Отчёт по верификации базовых формул:

```bash
python -m pippa.verification
```

Тесты:

```bash
pytest
```

## Деактивация окружения

```bash
deactivate
```
