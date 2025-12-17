#!/usr/bin/env python3
"""
Главный скрипт для запуска всех проверок расчетов Теории Pippa

Запускает все верификационные скрипты и выводит сводный отчет.
"""

import subprocess
import sys
import os

def run_script(script_name):
    """Запуск скрипта и возврат результатов"""
    print(f"\n{'='*60}")
    print(f"Запуск {script_name}")
    print('='*60)

    try:
        result = subprocess.run([sys.executable, script_name],
                              capture_output=True, text=True, timeout=30)
        print(result.stdout)
        if result.stderr:
            print("Ошибки:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Таймаут при выполнении {script_name}")
        return False
    except Exception as e:
        print(f"Ошибка при запуске {script_name}: {e}")
        return False

def main():
    """Главная функция"""
    print("=== Проверка всех расчетов Теории Pippa ===")
    print("Это займет несколько минут...")

    # Список скриптов для проверки
    scripts = [
        "fractal_dimension_verification.py",
        "particle_mass_verification.py",
        "fine_structure_constant_verification.py",
        "modes_equation_verification.py"
    ]

    # Результаты
    results = []

    for script in scripts:
        script_path = os.path.join(os.path.dirname(__file__), script)
        if os.path.exists(script_path):
            success = run_script(script_path)
            results.append((script, success))
        else:
            print(f"Скрипт {script} не найден")
            results.append((script, False))

    # Сводный отчет
    print(f"\n{'='*60}")
    print("СВОДНЫЙ ОТЧЕТ")
    print('='*60)

    successful = sum(1 for _, success in results if success)
    total = len(results)

    print(f"Успешно выполнено: {successful}/{total}")

    for script, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {script}")

    print("\n=== Рекомендации ===")
    if successful == total:
        print("✓ Все расчеты Теории Pippa проверены и корректны!")
        print("Теория математически последовательна.")
    elif successful >= total * 0.8:
        print("⚠ Большинство расчетов корректны, но есть небольшие расхождения.")
        print("Теория в целом последовательна, но требует уточнения параметров.")
    else:
        print("✗ Обнаружены серьезные проблемы в расчетах.")
        print("Теория требует фундаментальной переработки.")

    print("\n=== Следующие шаги ===")
    print("1. Проанализировать любые расхождения в расчетах")
    print("2. Уточнить параметры теории при необходимости")
    print("3. Дополнить документ исправленными формулами")
    print("4. Подготовить публикацию результатов")

if __name__ == "__main__":
    main()
