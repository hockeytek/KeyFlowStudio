# KeyFlow Studio - Architecture & Development Guide

## Актуальное состояние

KeyFlow Studio больше не является только экраном запуска одного matting-пайплайна. Текущая архитектура строится вокруг визуального node graph, централизованных контрактов нод и plan-driven runtime.

Главные документы по текущей системе:
- [docs/NODE_GRAPH_STANDARD.md](docs/NODE_GRAPH_STANDARD.md) — основной стандарт нод, совместимости и runtime-ожиданий
- [docs/plans/plan-nodeGraphStandardization.prompt.md](docs/plans/plan-nodeGraphStandardization.prompt.md) — компактный roadmap завершённой стандартизации
- [STAGE6_PROGRESS.md](STAGE6_PROGRESS.md) — история интеграционных этапов и тестового покрытия

## Структура проекта

```text
KeyFlow Studio/
├── main.py                              # Точка входа и MainWindow
├── README.md                            # Пользовательский обзор
├── QUICKSTART.md                        # Быстрый запуск
├── ARCHITECTURE.md                      # Этот файл
├── docs/
│   ├── NODE_GRAPH_STANDARD.md           # Основной стандарт node graph
│   └── plans/
│       └── plan-nodeGraphStandardization.prompt.md
├── app/
│   ├── coordinators/
│   │   ├── matting_orchestrator.py      # Сбор runtime-конфига и запуск графа/маттинга
│   │   ├── viewer_preview_controller.py # Маршрутизация предпросмотра
│   │   └── write_output_adapter.py      # Адаптер выходных write-нод
│   ├── node_graph/
│   │   ├── engine.py                    # Валидация графа, диагностика, execution plan
│   │   ├── models.py                    # GraphNode / GraphEdge
│   │   ├── specs/                       # NodeSpec: структура нод и портов
│   │   ├── rules/                       # NodeContract + registry
│   │   └── nodes/                       # Узкоспециализированные runtime/controller helper-ы
│   ├── services/                        # Доступ к моделям и backend inference
│   ├── workers/
│   │   └── inference_worker.py          # Исполнение подготовленного плана в фоне
│   ├── runtime_contract.py              # Production-safe runtime semantics
│   ├── node_graph_dialog.py             # Визуальный редактор графа, свойства, diagnostics
│   └── settings.py                      # QSettings и пользовательские настройки
└── tests/                               # Unit/smoke/integration coverage
```

## Архитектурные слои

```text
┌─────────────────────────────────────────────────────┐
│ UI Layer                                            │
│ main.py (MainWindow) + node_graph_dialog.py         │
│ Редактирование графа, запуск, preview, diagnostics  │
└──────────────────────┬──────────────────────────────┘
                       │
                       │ собирает конфиг выполнения
                       ↓
┌─────────────────────────────────────────────────────┐
│ Coordination Layer                                  │
│ matting_orchestrator.py                             │
│ Runtime config, export routing, cancel semantics    │
└──────────────────────┬──────────────────────────────┘
                       │
                       │ валидирует и планирует
                       ↓
┌─────────────────────────────────────────────────────┐
│ Graph Semantics Layer                               │
│ specs/ + rules/ + engine.py                         │
│ NodeSpec, NodeContract, registry, diagnostics, plan │
└──────────────────────┬──────────────────────────────┘
                       │
                       │ исполняет план в фоне
                       ↓
┌─────────────────────────────────────────────────────┐
│ Runtime Layer                                       │
│ inference_worker.py + services/                     │
│ Загрузка медиа, инференс, запись результатов        │
└──────────────────────┬──────────────────────────────┘
                       │
                       │ использует
                       ↓
┌─────────────────────────────────────────────────────┐
│ External Stack                                      │
│ PySide6, PyTorch, OpenCV, ffmpeg, model modules     │
└─────────────────────────────────────────────────────┘
```

## Основной поток выполнения

1. Пользователь собирает граф в UI.
2. [main.py](main.py) передаёт запуск в [app/coordinators/matting_orchestrator.py](app/coordinators/matting_orchestrator.py).
3. Оркестратор экспортирует graph preset и строит runtime config.
4. [app/node_graph/engine.py](app/node_graph/engine.py) валидирует граф и строит execution plan.
5. [app/workers/inference_worker.py](app/workers/inference_worker.py) исполняет план в фоновом потоке.
6. Сервисы моделей и write-адаптеры формируют выходные потоки.
7. UI получает progress, diagnostics и preview updates через Qt signals/slots.

## Ключевые компоненты

### MainWindow в [main.py](main.py)

Отвечает за:
- жизненный цикл приложения
- настройки устройства и среды
- запуск диалогов и оркестраторов
- приём сигналов progress/error/finished

### Graph UI в [app/node_graph_dialog.py](app/node_graph_dialog.py)

Отвечает за:
- визуальное редактирование графа
- свойства нод
- detached diagnostics dialog
- экспорт и импорт graph preset

### Оркестратор в [app/coordinators/matting_orchestrator.py](app/coordinators/matting_orchestrator.py)

Отвечает за:
- сбор write-таргетов и preview routing
- переключение между графовым и прямым runtime
- runtime config и cancel semantics

### Семантика графа в [app/node_graph/rules/node_contracts.py](app/node_graph/rules/node_contracts.py)
, [app/node_graph/rules/registry.py](app/node_graph/rules/registry.py) и [app/node_graph/engine.py](app/node_graph/engine.py)

Отвечает за:
- структуру портов и default properties
- топологические ограничения
- compatibility exceptions
- diagnostics и execution planning

### Runtime в [app/workers/inference_worker.py](app/workers/inference_worker.py)

Отвечает за:
- выполнение подготовленного плана
- вызов model services
- обработку ошибок и остановки
- запись промежуточных и финальных результатов

## Актуальные архитектурные принципы

1. Структура ноды задаётся через NodeSpec.
2. Семантика графа задаётся через NodeContract и registry.
3. Engine валидирует и планирует, но не исполняет.
4. Worker исполняет, но не придумывает семантику сам.
5. UI не должен обходить объявленные контракты нод.

## Расширение функциональности

## 🧩 Контракт интеграции нод (обязательно для всех будущих изменений)

Этот раздел фиксирует правила, по которым нужно добавлять/менять ноды, чтобы интеграция оставалась стабильной и предсказуемой.

### 1) Нода начинается со спецификации

Каждая новая нода сначала описывается в `app/node_graph/specs/*.py` через `NodeSpec`:

- `key` — стабильный идентификатор.
- `inputs`/`outputs` — порты и их типы данных.
- `default_properties` — полный набор дефолтных runtime/UI параметров.
- `title_i18n_key`/`subtitle_i18n_key` — только через i18n, без hardcode-строк.

После этого спецификация регистрируется в `app/node_graph/specs/__init__.py`.

### 2) UI свойств не должен ломать runtime-контракт

Если ноде нужны настройки, добавляется отдельная properties-панель (по аналогии с load/sam/matting/write).

Правило:

- Панель только читает/пишет `node.properties`.
- Бизнес-логика обработки не живет в panel-классе.
- Все надписи и тултипы идут через `app/i18n.py`.

### 3) NodeGraphDialog — UI-слой, не источник правил

В `app/node_graph_dialog.py` должно оставаться только UI-поведение (snap, hit-test, визуальная логика).
Семантические правила связей не должны дублироваться в диалоге.

Источник истины для правил:

- `app/node_graph/rules/node_contracts.py` — декларация контрактов нод.
- `app/node_graph/rules/registry.py` — API правил совместимости и topology.
- `app/node_graph/engine.py` — централизованная валидация + диагностика + план исполнения.

При добавлении новой ноды в UI обязательно проверить:

- Совместимость портов только через `NodeRulesRegistry`.
- Определение входного stream для Write (`_write_input_stream`), если нода может идти в Write.
- Обновление info-бейджа Write (`_refresh_write_panel_info`) при необходимости.
- Экспорт/импорт графа в пресеты через уже существующие `export_graph_preset` / `apply_graph_preset`.

Нельзя добавлять ветки, которые обходят сериализацию пресетов.

### 3.1) Diagnostics contract (обязательный)

`NodeGraphEngine` возвращает структурированные diagnostics (`GraphDiagnostic`) с полями:

- `code` (например `NG001` ... `NG011`)
- `node_id` / `src_node_id` / `dst_node_id`
- `src_port` / `dst_port`
- `rule`
- `message`

Требование:

- Новые проверки валидации добавляются как новый `code` + заполненный контекст.
- Ошибки валидации не должны выбрасываться «сырыми» строками без кода.

### 3.2) ExecutionPlan boundary (обязательный)

Между валидацией и исполнением всегда должен строиться `ExecutionPlan`.

`ExecutionPlan` включает:

- `execution_order`
- `node_actions` (`execute`, `passthrough_source`, `deferred`, `write_sink`, `skip_isolated`, `skip_disabled`)
- `deferred_node_ids`
- `deferred_corridorkey_sources`

Worker обязан выполнять граф по `node_actions`, а не по ad-hoc условиям в runtime-цикле.

### 4) Write-нода: source-agnostic принцип

Write должна сохранять payload, пришедший на `write.in`, независимо от источника.

Обязательные принципы:

- Нет жесткой привязки к конкретной upstream-ноде.
- Маршрутизация в runtime строится по метаданным источника (`source_node_type`, `source_port`, `stream`).
- Формат вывода определяется настройками Write, а не типом ноды-источника.

### 5) Runtime-роутинг должен быть расширяемым и plan-driven

В `main.py` для новых источников данных в Write:

- Добавлять отдельный небольшой save-handler (или переиспользовать универсальный).
- Не дублировать кодеки/форматы в нескольких местах.
- Для passthrough-сценариев не запускать Matting, если он не нужен графом.

В `app/workers/inference_worker.py`:

- Нельзя добавлять новую маршрутизацию исполнения через разрозненные `if node_type == ...` в основном цикле.
- Любая новая execution-ветка сначала описывается как `node_action` в `ExecutionPlan`, затем выполняется в worker.

### 6) i18n обязателен

Любая новая строка UI должна быть в `app/i18n.py` (ru/en).

Запрещено:

- Вшивать пользовательские строки прямо в `main.py`/панели.
- Добавлять только один язык.

### 7) Definition of Done для новой ноды

Изменение считается завершенным только если выполнено всё:

1. Спецификация добавлена и зарегистрирована.
2. i18n-ключи добавлены.
3. Свойства читаются/пишутся через `node.properties`.
4. Нода корректно сохраняется/восстанавливается в graph preset.
5. Проверены связи с Write (если применимо).
6. Нет ошибок в diagnostics (`main.py` и связанные файлы).
7. Пройден минимальный runtime smoke test сценария ноды.

### 8) Антипаттерны (что не делать)

- Делать special-case логику только под одну ноду, если это можно решить через контракт порта/потока.
- Хранить runtime-состояние ноды вне `properties`, если оно должно переживать preset export/import.
- Менять поведение Write так, чтобы оно зависело от "имени" ноды вместо фактического входного payload.

### Добавить параметр обработки

1. **В UI / properties panel:**
    - добавить элемент управления в соответствующую панель ноды или в окно настроек
    - сохранить значение в `node.properties`

2. **В orchestration/runtime:**
    - убедиться, что значение попадает в exported graph preset или runtime config
    - читать параметр в [app/workers/inference_worker.py](app/workers/inference_worker.py) или в специализированном runtime handler

### Добавить новый слой обработки

1. Создать или расширить `NodeSpec` и `NodeContract`.
2. При необходимости добавить model/service adapter в `app/services/`.
3. Подключить исполнение через [app/workers/inference_worker.py](app/workers/inference_worker.py).
4. Прокинуть preview/write semantics через orchestrator при необходимости.

### Поддержка интерактивной маски (клики)

Сейчас такая логика должна добавляться через существующие SAM-контроллеры и состояние runtime, а не через отдельный ad-hoc canvas в старом UI-слое.

## ⚙️ Технические детали

### Потокизация

- **UI thread:** MainWindow, отвечает за интерфейс
- **Worker thread:** InferenceWorker, выполняет тяжелые вычисления
- **Qt сигналы:** безопасная коммуникация между потоками

### Управление памятью

```python
# При загрузке видео 4K (3840x2160) на 300 кадров:
# ≈ 320 MB для кадров + 320 MB для модели = ≈ 640 MB
# На моделях Apple Silicon это норма

# Для очень больших видео добавить:
# - Потоковая обработка (буффер N кадров)
# - Resize больших видео перед обработкой
```

### Device selection (app/utils/device.py)

```python
if torch.cuda.is_available():
    device = "cuda"        # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"         # Apple Silicon (M1/M2/M3)
else:
    device = "cpu"         # Fallback
```

## 🐛 Отладка

### Включить логирование

Добавить в main.py:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Проверить какой device используется

```bash
python -c "from app.utils import get_device_name; print(get_device_name())"
```

### Посмотреть предупреждения PyTorch

```bash
export PYTHONWARNINGS=default
python main.py
```

## 📦 Публикация

### Локальное тестирование

```bash
python main.py
```

### Для распространения (не требуется для личного использования):
- Code signing
- Notarization (для macOS Catalina+)
- Disk image (.dmg)

---

**Проект готов к расширению и модификации. При вопросах добавьте логирование и проверьте каждый шаг потока данных.**
