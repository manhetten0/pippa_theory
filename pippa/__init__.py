"""Pippa theory verification framework.

Модульный фреймворк для проверки правильности расчётов формул
теории Pippa и проведения симуляций.

Структура:
- constants:          физические и теоретические константы
- fractal_dimension:  численный вывод фрактальной размерности D = 4/pi
- particle_physics:   константы связи и массы частиц/бозонов
- gravity:            эмерджентная гравитация (Приложение G)
- dark_matter:        профиль тёмной материи и закон масштабирования
- verification:       сравнение предсказаний с экспериментом

Реализованы БАЗОВЫЕ (точные) формулы. Степенные/феноменологические
формулы (массы лептонов с beta, нейтрино) помечены как неточные
и вынесены в particle_physics.phenomenology.
"""

from . import constants
from . import fractal_dimension
from . import particle_physics
from . import gravity
from . import dark_matter
from . import verification

__all__ = [
    "constants",
    "fractal_dimension",
    "particle_physics",
    "gravity",
    "dark_matter",
    "verification",
]

__version__ = "0.1.0"
