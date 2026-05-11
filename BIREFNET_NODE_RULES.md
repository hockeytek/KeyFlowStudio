# Правила Подключения Нод К BiRefNet

## Назначение

Этот документ фиксирует правила для ноды BiRefNet и разделяет:

- факты из оригинального GitHub-проекта авторов
- наши локальные правила интеграции в KeyFlow Studio

Цель: убрать путаницу между исходным RGB, alpha-hint и финальным результатом downstream-нод.

## Источники (Проверено по GitHub Авторов)

Базовый репозиторий авторов:

- https://github.com/ZhengPeng7/BiRefNet

Проверенные файлы:

- README.md
- inference.py
- config.py

Ключевые подтвержденные факты из авторского репо:

- BiRefNet позиционируется как модель для dichotomous image segmentation.
- В инференсе авторы получают карту предсказания через sigmoid и сохраняют ее как mask/prediction map.
- В model zoo у авторов есть пресеты для разных задач: general use, matting, portrait, DIS, HRSOD, COD.
- Авторы явно выделяют сценарии matting и general use как отдельные направления.

## Что Является Контрактом В Нашем Проекте (Локальная Интеграция)

По нашей нодовой спецификации:

- Вход BiRefNet:
  - `Image` (тип `image`, required)
- Выход BiRefNet:
  - `Alpha` (тип `alpha`)

Семантика в KeyFlow Studio:

- `Image` = RGB-кадр/последовательность кадров
- `Alpha` = одноканальная alpha/mask карта (hint)

Важно: это контракт именно нашей node-обвязки, а не цитата 1:1 из README авторов.

## Правила Подключения Входа BiRefNet

### Вход `Image`

- Принимает только `image`-поток
- Должен быть исходным визуальным RGB-входом

Рекомендуемые источники:

- Source
- Load
- любая upstream-нода с корректным RGB output

Не подключать:

- alpha/mask как основной image-вход
- другие масочные данные, если порт семантически не image

## Правила Выхода BiRefNet

### Выход `Alpha`

- Это alpha/mask-поток для downstream-задач
- Это не финальный compositing-ready RGBA output

В нашем пайплайне основной сценарий:

- `BiRefNet.Alpha -> CorridorKey.Alpha Hint`

Дополнительно:

- `BiRefNet.Alpha -> Write` для сохранения промежуточного hint/mask

## Длина Последовательностей

Локальное правило интеграции:

- Если на вход подана последовательность кадров, BiRefNet должен отдавать `Alpha` той же длины
- Если вход один кадр, выход тоже один кадр

Это правило критично для downstream-ноды CorridorKey:

- длина `Alpha Hint` должна совпадать с длиной входного `Image` у CorridorKey;
- при несовпадении CorridorKey останавливает запуск с ошибкой валидации.

## Связка BiRefNet С CorridorKey (Staged Workflow)

Для видео-сценариев в KeyFlow Studio базовый рабочий режим связки BiRefNet -> CorridorKey выполняется последовательно.

Порядок выполнения:

1. BiRefNet обрабатывает кадры диапазона и формирует alpha-hint секвенцию.
2. Alpha-hint сохраняется как последовательность масок.
3. BiRefNet выгружается из памяти.
4. CorridorKey запускается с уже подготовленным `Alpha Hint` и основным `Image`.

Почему так:

- BiRefNet и CorridorKey используют тяжелые веса моделей;
- одновременное удержание обеих моделей в памяти ухудшает производительность и стабильность;
- staged-процесс снижает расход RAM/VRAM и делает pipeline предсказуемым.

## Локальные Параметры Ноды BiRefNet

Поддерживаемые свойства в KeyFlow Studio:

- `usage` (профиль модели)
- `half_precision`
- `dilate_radius`
- `erode_radius`

Локальная постобработка (не утверждение из авторского README):

- в worker применяется пороговая бинаризация alpha перед morphology (dilate/erode)
- это сделано для стабильного coarse hint в связке с CorridorKey в нашем проекте

## Рекомендации Для Write При `BiRefNet.Alpha`

Для технического хранения hint/mask:

- `EXR`
- `PNG` (при необходимости 16-bit)

Для быстрого просмотра:

- `PNG` 8-bit
- video-контейнеры только как preview-артефакт

## Роль BiRefNet В Связке С CorridorKey (Локальная Архитектура)

Рекомендуемая схема:

- `Source/Load -> BiRefNet.Image`
- `BiRefNet.Alpha -> CorridorKey.Alpha Hint`
- `Source/Load -> CorridorKey.Image`

Смысл:

- BiRefNet готовит coarse hint
- CorridorKey использует hint для последующего уточнения результата
- в видео-режиме связка выполняется последовательно (staged): сначала BiRefNet, затем CorridorKey

## Что Нельзя Путать

- `BiRefNet.Alpha` не равен `CorridorKey.Processed`
- `BiRefNet.Alpha` не гарантированно финальный production matte во всех сценариях
- mask/alpha поток нельзя автоматически трактовать как image-вход

## Краткая Памятка

- `BiRefNet.Image` = исходный RGB
- `BiRefNet.Alpha` = alpha-hint / mask
- Базовый downstream в нашем графе: `BiRefNet.Alpha -> CorridorKey.Alpha Hint`
- Для CorridorKey обязательно совпадение количества кадров `Alpha Hint` и `Image`
- Рекомендуемый процесс для видео: сначала BiRefNet, потом CorridorKey
