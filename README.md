# Модель для распознавания именованных сущностей
Осннована на [этой](https://arxiv.org/pdf/1511.08308v5.pdf) работе
#Отличия от модели из работы:
-Embedding-и тренируются (в оригинале - предтренированные)
-Другие значаения гиперпараметров
#Результаты
Валидационная точность во время тренировки высокая - 95+% в зависимости от гиперпараметров.
Модель справляется хорошо в случае грамматически правильного текста. Если в тексте есть ошибки,
модель начинает показывать себя плохо (например, не распознаёт имена, написанные со строчной буквы)
