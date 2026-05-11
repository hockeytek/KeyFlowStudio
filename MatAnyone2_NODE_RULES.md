# Правила Подключения Нод К MatAnyone2

## Назначение

Этот документ фиксирует правила для ноды MatAnyone2 и разделяет:

- подтвержденное поведение из оригинального авторского контекста MatAnyone2
- наши локальные правила интеграции в KeyFlow Studio

Цель документа: исключить путаницу между входной seed-mask, выходным alpha matte, RGB-выходом `fg`, preview-результатами и production downstream-подключениями.

## Источники

Первичный источник по поведению MatAnyone2:

- официальный GitHub-репозиторий авторов MatAnyone2
- README, CLI и released inference code авторов (`inference_matanyone2.py`, `matanyone2/cli.py`, `matanyone2/inference/inference_core.py`)

Локальный контракт KeyFlow Studio подтвержден по текущей реализации:

- `app/node_graph/specs/matting.py`
- `app/node_graph/matting_properties_panel.py`
- `app/node_graph/nodes/matting_controller.py`
- `app/workers/inference_worker.py`
- `app/services/model_service.py`
- `app/node_graph/engine.py`
- `app/node_graph_dialog.py`
- `app/i18n.py`

Важно:

- данный документ не заменяет оригинальную документацию авторов
- при расхождении приоритет у авторского репозитория и авторского inference/API-кода
- оригинальный репозиторий MatAnyone2 не описывает нодовый интерфейс; имена портов `img`, `mask`, `fg`, `alpha` являются локальным отображением авторской семантики в KeyFlow Studio

## Что Подтверждено По Авторскому Контексту MatAnyone2

### Что подтверждено

В авторском репозитории MatAnyone2 подтвержден не нодовый, а inference-контракт:

- входной `input_path`: видеофайл или папка кадров
- входной `mask_path`: first-frame segmentation mask
- optional `ckpt_path`: путь к checkpoint/весам MatAnyone2 (CLI `--ckpt-path`)
- runtime-параметры warmup / erode / dilate
- выход `foreground output`
- выход `alpha output`

Уточнение по путям весов в KeyFlow Studio:

- upstream CLI MatAnyone2 может показывать свой локальный default-путь для checkpoint
- локальный runtime KeyFlow Studio не использует project-local fallback-папки как обязательный источник
- в локальной интеграции веса кэшируются в platform app-data models dir (или в `KEYFLOW_MODELS_DIR`, если задан)

Для batch / CLI / Python API подтверждено следующее:

- один запуск MatAnyone2 принимает видео или sequence кадров
- один запуск MatAnyone2 принимает одну first-frame segmentation mask
- результаты сохраняются как foreground output и alpha output
- при `save_image` авторский код сохраняет покадровые папки `fgr` и `pha`

Ключевой факт по семантике author output:

- в released inference code авторов `foreground output` собирается как RGB-композит `source_rgb * alpha + background * (1 - alpha)`
- в released inference code фоновый цвет для такого `foreground output` зеленый
- следовательно, author `foreground output` не является прозрачным RGBA-результатом

Подтверждено частично:

- авторская документация подтверждает режим `video/frames + first-frame mask`
- авторская документация не вводит понятия нодовых портов, `preview`, `processed result`, downstream `Write`, port typing или локальных правил совместимости `mask/alpha`

### Что не подтверждено по авторскому контексту

В авторской документации не подтверждено как базовый публичный контракт:

- отдельный нодовый вход для per-frame correction masks
- полноценный вход в виде независимой mask-sequence той же длины, что и video
- отдельный `processed` / `RGBA` output
- downstream-правила сохранения в EXR / PNG 16-bit / ProRes 4444
- локальный checker-background режим

Все такие правила ниже относятся только к интеграции KeyFlow Studio.

## Что Является Контрактом В Нашем Проекте (Локальная Интеграция)

Локальный нодовый контракт KeyFlow Studio для MatAnyone2:

- вход `img` имеет тип `image`
- вход `mask` имеет тип `mask`
- выход `fg` имеет тип `image`
- выход `alpha` имеет тип `alpha`

Локальная семантика портов:

- `img` = исходный RGB image / frame sequence
- `mask` = seed-mask / guide-mask для запуска маттинга; это не финальный matte
- `alpha` = предсказанный matte / alpha output
- `fg` = RGB-результат с уже подложенным фоном, а не прозрачный foreground

Локальные правила интеграции:

- в графовом движке KeyFlow Studio типы `mask` и `alpha` считаются совместимыми для соединений
- `Write.in` локально принимает любой upstream stream, даже если в spec он формально объявлен как `image`
- локальная UI-настройка `fg_background` меняет фон для `fg`-выхода между `green` и `checker`
- checker-background является только локальным режимом KeyFlow Studio, а не авторским контрактом MatAnyone2
- у ноды MatAnyone2 в текущей интеграции нет отдельного output-порта `processed`
- у ноды MatAnyone2 в текущей интеграции нет отдельного output-порта `RGBA`

Что считается preview в нашем проекте:

- runtime preview в viewer
- runtime preview на ноде Write
- `fg` в случаях, когда нужен быстрый визуальный контроль результата на подложке
- alpha, записанный в обычный video-контейнер, если он используется только для просмотра

Что считается production result в нашем проекте:

- `alpha`, когда он сохранен в формат, сохраняющий корректную matte-семантику
- `fg`, только если downstream действительно ожидает baked RGB-result без прозрачности

Что не считается production-safe final processed result:

- `fg`, если downstream ожидает RGBA с alpha-каналом
- любой preview-only video от alpha stream
- визуально правдоподобный `fg`, ошибочно принятый за transparent foreground

Дополнительное локальное замечание по реализации:

- legacy handler `app/node_graph/nodes/matting_node.py` удален; dual-path исполнения больше не является контрактом
- активный run-path текущего приложения для графов с MatAnyone2 идет через `InferenceWorker` node-graph execution и производит потоки `fg` и `alpha`
- для downstream-семантики authoritative считается только активный worker/service путь

## Phase 3: Единый Runtime Contract

В проекте зафиксирован единый runtime contract в `app/runtime_contract.py`.

Единая схема входов (`RuntimeConfig`):

- `is_video: bool`
- `start_frame: int`
- `end_frame: int`
- `compatibility_profile: auto|legacy_intel|apple_silicon`
- `correction_masks: dict[int, mask]` (optional)
- `node_graph: {nodes, edges}` для graph execution path
- `fg_write` / `alpha_write` (optional, для write-политик)

Единая схема выходов (`RuntimeResult`):

- `status: ok|cancelled|error`
- `cancelled: bool`
- `saved_paths: dict[node_id, path]`
- `n_frames: int`
- optional legacy fallback: `fgr_path`, `alpha_path`

Единая схема progress:

- stage progress нормализуется через `normalize_stage_progress(percent, status_text)`
- frame progress нормализуется через `normalize_frame_progress(current, total)`
- диапазон percent всегда `[0..100]`

Единая cancel-семантика:

- отмена фиксируется как `status=cancelled` и `cancelled=true`
- проверка отмены в orchestration/controller идет через `is_runtime_cancelled(...)`

Единая write/preview семантика (`GraphStreamPreviewPayload`):

- `semantics=preview_only`: кадр валиден только для UI-preview, путь не считается production-safe артефактом
- `semantics=production_safe`: путь можно регистрировать как итог write-output и использовать в downstream
- правило для UI: `preview_only` не должен автоматически становиться persisted final output path

## Phase 4: i18n и Annotation/Status Consistency

Для MatAnyone2 runtime-контекста используется отдельный namespace ключей статуса:

- `matting_wait_processing`
- `matting_status_start`
- `matting_status_cancel`
- `matting_status_frame`
- `matting_status_stopped`
- `matting_status_done`
- `matting_status_error`

Правило интеграции:

- SAM-ключи (`sam_*`) не используются как primary-ключи в MatAnyone2 status/annotation слое
- допускается только fallback на нейтральные ключи `status_*` для обратной совместимости

## Phase 5: UX Cancel/Save Semantics

В MatAnyone2 runtime введена явная политика отмены `cancel_policy`:

- `immediate`: быстрая остановка без сохранения partial outputs

Решение по UX для кнопки Stop:

- Stop в MatAnyone2 трактуется только как `immediate`
- partial-save/cleanup режимы не применяются для действия кнопки Stop
- UI-статус после Stop: `matting_status_stopped`

Engineering-note:

- coordinator для действия Stop принудительно выставляет `cancel_policy=immediate`

## Правила Для Входов MatAnyone2

### Вход `img`

- Тип: `image`
- Семантика: исходный RGB-поток, который подается в MatAnyone2 как video / sequence / single image

Можно подключать:

- `Source`
- `Load`
- любой upstream-узел с корректным RGB image output

Нельзя подключать:

- `alpha`
- `mask`
- SAM2 mask
- любой single-channel matte / alpha / guide-mask поток

Важно не путать:

- `img` это source RGB
- `img` не является mask input
- `img` не является alpha input

### Вход `mask`

- Тип: `mask`
- Семантика: seed / guide mask для запуска маттинга

Что подтверждено по author semantics:

- это first-frame segmentation mask
- это не финальный предсказанный matte

Что является локальным правилом KeyFlow Studio:

- движок разрешает сюда подключать как `mask`, так и `alpha`-потоки
- SAM-output может использоваться как допустимый upstream для этого входа
- если на `mask` приходит последовательность масок (например, из `Alpha` node), MatAnyone2 берет только первый кадр как seed-mask; остальные кадры входной mask-sequence игнорируются

Можно подключать:

- `SAM` mask
- `Alpha` node
- любой upstream-узел с корректным mask / alpha output

Нельзя подключать:

- исходный RGB
- `fg`
- `comp`
- `processed`
- любой полноцветный preview вместо маски

Важно не путать:

- вход `mask` это guide / seed input
- вход `mask` не равен выходу `alpha`
- подключение `alpha` сюда допустимо только как локальная совместимость KeyFlow Studio, а не как author node contract
- `mask`-последовательность не является per-frame управляющим входом MatAnyone2 в текущем контракте

## Правила Для Длины Последовательностей

Что подтверждено по авторскому контексту:

- базовый режим MatAnyone2: один video / frame sequence + одна first-frame segmentation mask
- растягивание одной seed-mask на всю последовательность является базовым режимом MatAnyone2, а не исключением

Что является контрактом в нашем проекте:

- если `img` это video / sequence, одна подключенная `mask` используется как базовый seed для всего run
- если `img` это single image, `mask` тоже должна быть single image / single mask
- отдельный публичный input-порт для полноценной mask-sequence той же длины не предусмотрен
- дополнительные per-frame correction masks могут появляться только через локальный SAM-side-channel KeyFlow Studio
- такие correction masks не меняют базовый публичный контракт ноды MatAnyone2

Важно:

- полноценная независимая mask-sequence как основной публичный режим для MatAnyone2 здесь не документируется как подтвержденный author fact
- если downstream-сценарий требует отдельную маску на каждый кадр как базовый контракт, это уже не базовая семантика текущей MatAnyone2-ноды

## Правила Для Выходов MatAnyone2

### Выход `alpha`

- означает: предсказанный alpha matte / matte stream
- базовая семантика: single-channel alpha result
- внутренне может существовать как float alpha; при записи формат вывода определяет точность

Чем этот выход не является:

- не является входной seed-mask
- не является RGB
- не является `fg`
- не является финальным RGBA-изображением

Как правильно использовать downstream:

- подключать туда, где downstream ожидает matte / alpha
- подключать в `Write` для сохранения matte
- использовать как масочный / alpha stream в локальных цепочках, где допустима mask/alpha-совместимость

### Выход `fg`

- означает: RGB-результат MatAnyone2 с уже подложенным фоном
- в author inference code это RGB-композит исходного изображения через alpha на цветной фон
- в локальной интеграции фон выбирается через `fg_background`

Чем этот выход не является:

- не является matte
- не является transparent foreground
- не является RGBA
- не является `processed` output
- не является final alpha-bearing result

Как правильно использовать downstream:

- подключать туда, где downstream ожидает обычный RGB image stream
- использовать для review, editorial, approval, quick preview
- использовать как delivery RGB только если baked background является ожидаемым результатом

## Рекомендации Для Write / Downstream Usage

### Выход `alpha`

Рекомендуется:

- `EXR`, если нужен production-safe matte с максимальной точностью
- `PNG` 16-bit, если нужен дисковый matte в image-sequence без EXR

Допустимо:

- `PNG` 8-bit, если задача не требует точной matte-градации

Только preview-only:

- `MP4`
- `MOV`
- другой обычный video-контейнер

Важно не путать:

- video-запись `alpha` не превращает его в настоящий alpha-bearing container
- для video-кодеков локальная реализация продвигает single-channel alpha в 3-канальный grayscale-preview
- это удобно для просмотра, но это не production matte master
- `JPG` для `alpha` использовать не следует

### Выход `fg`

Рекомендуется:

- `PNG`, если нужен обычный RGB still / image sequence
- `EXR`, если нужен high-quality RGB output без потерь, но все еще без отдельного alpha-порта в самом файле по контракту этой ноды
- video-форматы, если downstream ожидает обычный RGB delivery

Допустимо:

- `MP4`
- `MOV`
- `ProRes`, если нужен качественный RGB-video-output

Важно не путать:

- `ProRes 4444` на одном только `fg`-stream не делает этот поток автоматически `RGBA`
- если нужен настоящий final RGBA, его нельзя объявлять выходом самой ноды MatAnyone2 без отдельного подтвержденного local contract
- в текущей интеграции у MatAnyone2-ноды нет отдельного production-safe `processed` порта

### Preview-only output

Preview-only output в текущем проекте:

- runtime thumbnails на Write
- viewer previews
- `fg`, когда он используется только для визуальной проверки на green / checker background
- alpha, сохраненный в video-формат только ради просмотра

### Production-safe output

Production-safe output в текущем проекте:

- `alpha` в `EXR`
- `alpha` в `PNG` 16-bit
- `fg` в `EXR` / `PNG`, если downstream действительно нужен baked RGB без alpha

## Что Нельзя Путать

- вход `mask` и выход `alpha` это разные сущности
- `mask` это seed / guide input, а не финальный matte
- `alpha` это matte, а не RGB
- `fg` визуально может выглядеть как вырезанный foreground, но семантически это RGB с уже подложенным фоном
- `fg` не является matte
- `fg` не является RGBA
- `fg` не является final processed result с alpha-каналом
- у ноды MatAnyone2 нет отдельного `processed` output в текущем локальном контракте
- grayscale-video, записанный из `alpha`, не является тем же самым, что файл с настоящим alpha-каналом
- возможность подключать `alpha` в `mask` и возможность вести любой stream в `Write.in` это локальные правила KeyFlow Studio, а не author behavior MatAnyone2

## Краткая Памятка

- `img` = исходный RGB / video / frame sequence
- `mask` = first-frame seed-mask / guide-mask
- `alpha` = предсказанный matte / alpha output
- `fg` = RGB-результат с подложенным фоном
- `preview` = viewer / runtime thumbnail / video-preview matte
- `production-safe matte` = `alpha` в `EXR` или `PNG` 16-bit
- `final RGBA` = не является прямым output ноды MatAnyone2 в текущей интеграции