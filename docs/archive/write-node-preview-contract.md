# Write Node Preview Contract

## Цель

Для цепочки Source -> GVM -> Write нода Write должна:

1. Сохранять результат по вычисленному output path.
2. Читать preview из этого же output path.
3. Показывать этот же результат:
   - в thumbnail самой ноды Write;
   - в правом окне просмотра.

Если runtime-сигнал preview потерян или приложение было перезапущено, preview должен восстанавливаться с диска.

## Главный инвариант

Путь, по которому Write пишет результат, и путь, по которому UI потом ищет результат для preview, должны совпадать.

Нельзя держать preview только на временных runtime-сигналах.

## Production format policy

Для production-пайплайна Write должен считать основными форматами только:

- PNG
- EXR
- JPG
- MP4
- MOV

Режим `source` не должен продвигать legacy-форматы вроде WEBP/TIFF/BMP/AVI/MKV/WEBM/M4V как итоговый production output. Если вход пришел в одном из таких форматов, Write должен резолвить его в production-safe fallback:

- image legacy input -> PNG
- video legacy input -> MP4

Явно сохраненные старые пресеты с такими форматами можно поддерживать как compatibility path, но они не должны становиться частью основной матрицы форматов, UI guidance или restore-contract.

## Канонический путь Write

Write больше не живет только по legacy-схеме `base/stream`.

Канонический graph-aware путь строится через `build_graph_write_output_dir(...)` и включает:

- базовую output directory;
- title upstream-ноды;
- label upstream-порта.

Именно этот путь должен считаться источником истины для disk restore.

## Что обязана отдавать нода/диалог

`connected_write_targets()` должен возвращать достаточно данных, чтобы восстановить реальный output path Write:

- `graph_node_id`
- `stream`
- `source_node_type`
- `source_path`
- `source_port`
- `source_node_title`
- `port_label`
- `output_format`
- `auto_output_dir`
- `output_dir`
- `resolved_output_dir`
- `file_name`
- `last_output_path`

Без этого UI будет знать только stream, но не фактический каталог, в который writer реально сохранил файлы.

## Порядок restore с диска

`MattingOrchestrator._find_existing_write_output_path()` должен искать результат в таком порядке:

1. `last_output_path`, если файл существует.
2. `resolved_output_dir`, если он известен.
3. Fallback-путь, заново собранный через `build_graph_write_output_dir(...)`.

Restore не должен опираться только на legacy-схему `base/stream`.

## Поведение при выборе Write-ноды

При выборе export/Write-ноды viewer не должен зависеть только от `saved_output_path_for_node(node_id)`.

Если runtime-кэш пуст, нужно:

1. взять `last_output_path`, если он валиден;
2. иначе найти target через `connected_write_targets()`;
3. вычислить реальный output path;
4. применить его как preview path;
5. отрисовать preview в viewer и thumbnail ноды.

## Cloud path

Для cloud completion path обязательно пробрасывать вместе:

- `result_path`
- `write_node_id`

После завершения cloud job UI обязан:

1. сохранить final path для Write-ноды;
2. обновить thumbnail ноды;
3. обновить правый viewer.

## Симптом регрессии

Признак, что контракт снова сломан:

- файлы лежат в каталоге Write;
- но thumbnail Write-ноды пустой;
- и правый viewer тоже пустой или не подхватывает результат до нового runtime-сигнала.

## Файлы, где лежит логика

- `app/workers/cloud_inference_worker.py`
- `main.py`
- `app/node_graph_dialog.py`
- `app/coordinators/matting_orchestrator.py`
- `app/coordinators/viewer_preview_controller.py`

## Тесты, которые должны защищать поведение

- `tests/test_matting_orchestrator.py`
- `tests/test_smoke_sam2_tracking.py`

При будущих изменениях Write-ноды сначала проверять, что запись и restore используют один и тот же path contract.