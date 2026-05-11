# Правила Подключения Нод К CorridorKey

## Назначение

Этот документ фиксирует правила для ноды CorridorKey и разделяет:

- факты из оригинального репозитория авторов CorridorKey
- наши локальные правила интеграции в KeyFlow Studio

Цель: исключить путаницу между исходным RGB, alpha-hint, matte, preview-композитом и финальным RGBA-результатом.

## Источники

Первичный источник по поведению CorridorKey:

- официальный GitHub-репозиторий авторов CorridorKey

Важно:

- данный документ не заменяет оригинальную документацию авторов
- при расхождении приоритет у поведения из авторского репозитория

## Что Подтверждено По Авторскому Контексту CorridorKey

Для CorridorKey используются два разных семантических входа:

- `Image`: основной RGB image/frame sequence
- `Alpha Hint`: одноканальный hint (mask/matte/coarse alpha)

И набор выходов с разной семантикой:

- `Alpha`
- `FG`
- `Comp`
- `Processed`

Актуальный upstream CorridorKey поддерживает выбор цвета экрана:

- `green` — классический зеленый checkpoint `CorridorKey_v1.0.safetensors`
- `blue` — отдельный blue-screen checkpoint `CorridorKeyBlue_1.0.safetensors`
- `auto` — определение green/blue по первому кадру и alpha hint

В KeyFlow Studio default остается `green`, чтобы старые проекты и сохраненные графы не меняли результат после обновления. Blue-screen режим требует свежий Torch backend CorridorKey; upstream MLX blue пока не поддерживает.

## Что Является Контрактом В Нашем Проекте (Локальная Интеграция)

Локальный нодовый контракт в KeyFlow Studio:

- Вход `Image` имеет тип `image`
- Вход `Alpha Hint` имеет масочную семантику (mask/alpha hint)

Локальные выходы ноды:

- `Alpha` (matte)
- `FG` (foreground RGB)
- `Comp` (preview composite)
- `Processed` (final RGBA в текущей реализации пайплайна)

Важно: формулировки про типы и поведение выше относятся к нашей интеграции и не являются дословной цитатой README авторов.

## Правила Для Входов CorridorKey

### Вход `Image`

- Тип: `image`
- Семантика: исходный RGB поток

Можно подключать:

- Source
- Load
- любой upstream узел с корректным RGB image output

Нельзя подключать:

- `Alpha`
- SAM2 mask
- BiRefNet alpha
- любые single-channel mask/matte данные

### Вход `Alpha Hint`

- Семантика: одноканальная маска-подсказка
- Типовой контракт в рантайме: принимает только `mask` и `alpha` payload
- Источник: может быть любая нода, если ее выходной порт отдает `mask` или `alpha`

Можно подключать:

- масочные Read-потоки (`Alpha` node)
- BiRefNet `Alpha`
- SAM `out` (`alpha`)
- ChromaKey `mask`
- любые другие ноды с корректным `mask/alpha` output

Нельзя подключать:

- Source/Load `out` (RGB `image`)
- `FG`
- `Comp`
- `Processed`
- обычный полноцветный RGB preview

Примечание по интеграции KeyFlow Studio:

- Для `corridorkey.alphahint` в реестре действует порт-специфичное исключение топологии:
	источник не ограничивается списком `upstream_allowed` ноды CorridorKey.
- При этом проверка типов остается строгой: только `mask/alpha`.

## Правила Для Длины Последовательностей

Локальное правило KeyFlow Studio:

- если `Image` это sequence/video, `Alpha Hint` должен совпадать по длине
- если `Image` это одиночное изображение, `Alpha Hint` тоже должен быть одиночным

Автоматическое растягивание одной маски на всю последовательность не считается базовым режимом.

## Связка CorridorKey С SAM2 Mask (SAM2)

Рабочий контракт для видео-сценария:

1. Нода SAM2 Mask должна быть переведена в режим SAM2.
2. В интерфейсе должен быть задан диапазон кадров обработки.
3. В SAM2 должна быть подготовлена хотя бы одна опорная маска для старта трекинга.
4. SAM2 должен сформировать масочную секвенцию на весь выбранный диапазон.
5. После формирования секвенции SAM2 выгружается из памяти, затем запускается CorridorKey.

Почему это важно:

- И SAM2, и CorridorKey используют тяжелые веса.
- Одновременное удержание обеих моделей в памяти ухудшает производительность.
- Поэтому в связке применяется последовательный staged-процесс: сначала SAM2, затем CorridorKey.

Это согласовано с тем же принципом staged-выполнения, который используется в связке BiRefNet -> CorridorKey.

Проверка перед стартом CorridorKey:

- число кадров в Alpha Hint должно совпадать с числом кадров во входе Image;
- при несовпадении CorridorKey останавливает запуск с ошибкой валидации.

## Правила Для Выходов CorridorKey

### Выход `Alpha`

- matte / alpha результат
- не является RGB preview

### Выход `FG`

- foreground RGB
- не является alpha-маской

### Выход `Comp`

- preview composite
- служит для визуальной проверки

### Выход `Processed`

- финальный RGBA-поток в нашем пайплайне
- ориентирован на downstream compositing-задачи

## Рекомендации Для Write

### Если подключен `Alpha`

Рекомендуется:

- `EXR`
- `PNG` 16-bit

Video-контейнеры допустимы только как preview-артефакт.

### Если подключен `FG`

Допустимо:

- `EXR`
- `PNG`
- video форматы

### Если подключен `Comp`

Рекомендуется:

- `PNG`
- video форматы

### Если подключен `Processed`

Рекомендуется:

- `EXR`
- `ProRes 4444`, если нужен видеоформат с alpha

## Что Нельзя Путать

- `Alpha` и `FG` это разные сущности
- `Alpha` и `Processed` это разные уровни результата
- `Comp` это preview, не production matte
- визуально ч/б картинка не обязана быть корректным matte без проверки семантики порта

## Краткая Памятка

- `Image` = исходный RGB
- `Alpha Hint` = mask/hint
- `Alpha` = matte
- `FG` = foreground RGB
- `Comp` = preview composite
- `Processed` = final RGBA (в нашей интеграции)

---

## Параметры Ноды CorridorKey (актуальные, 2025–2026)

### `input_colorspace` (auto | srgb | linear)

Определяет, как интерпретируются входные кадры перед подачей в модель.

| Значение | Поведение |
|----------|-----------|
| `auto`   | Автоопределение: `dtype=float32 AND ext=.exr` → linear; иначе → sRGB |
| `srgb`   | Обычное видео/PNG/JPG в sRGB |
| `linear` | Линейные EXR/CG-рендеры (float32 0-1) |

**Техническая цепочка:**
- Одиночный EXR: `load_image_float` → `float32 linear [0..1]` → режим `linear` корректен.
- EXR-секвенция: `load_rgb_image` → `_ensure_uint8_rgb` (gamma 2.2) → `uint8` → режим `srgb` корректен.
- Авто-детект: условие `_is_float_frame AND _source_ext==".exr"` — **оба** критерия обязательны.

**Важно:** передача EXR-кадра с `input_is_linear=False` — баг: движок не конвертирует linear→sRGB перед нормализацией, что даёт неверные цвета и плохой кей.

---

### `hint_dilate_radius` (int, 0–100, default: 0)

Расширение маски-подсказки (BiRefNet/SAM) **перед подачей в CorridorKey** — отдельный шаг, независимый от встроенного `dilate_radius` ноды BiRefNet.

**Зачем:** чуть расширенная маска даёт модели больше контекста на краях → лучший кей на размытых/тонких деталях.

**Реализация:**
- Читается из `node.properties["hint_dilate_radius"]` в `_execute_corridorkey_node`.
- Применяется ко **всем трём** режимам загрузки маски (staged-disk, SAM-disk, batch in-memory) сразу после загрузки `alpha_hint`, **перед** вызовом `corridorkey_service.process_frame()`.
- Использует тот же метод `_apply_birefnet_mask_morphology(mask, dilate, erode=0)` что и BiRefNet.
- При `hint_dilate_radius == 0` шаг пропускается целиком.

**Отличие от BiRefNet `dilate_radius`:**
- `birefnet.dilate_radius` — применяется при генерации маски в BiRefNet-ноде (Stage 1 staged mode).
- `corridorkey.hint_dilate_radius` — применяется непосредственно перед inference CorridorKey (Stage 3).

---

### Удалённый легаси-код (2025)

В рамках очистки удалены:
- Ветка `hasattr(engine, "process")` — старое API `engine.process()`. Текущий движок всегда имеет только `process_frame()`.
- Fallback-ключи `rgba` и `straight` в выходном dict `corridorkey_service.process_frame()`.
- `import inspect` из `corridorkey_service.py`.
- `self._compatibility_profile` из `inference_worker.py` (переменная, которая определяла устаревшие preset-derived флаги).

Актуальный API `corridorkey_service.process_frame()`:
```python
process_frame(
    image,            # [H,W,3] float32 0-1
    alpha_hint,       # [H,W] float32 0-1 (2D)
    despill_strength, # 0-1 UI/runtime value
    despeckle,        # bool
    despeckle_size,   # int
    refiner_strength, # float 0-2
    use_refiner,      # bool
    input_is_linear,  # bool
)
# → dict: {alpha, fg, comp, processed}
```