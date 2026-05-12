# KeyFlow Studio

Qt-приложение для node-based видео keying, matting, mask generation и compositing workflow на базе MatAnyone2, SAM, BiRefNet, GVM, CorridorKey и связанных сервисов.

Это публичная preview-версия KeyFlow Studio. Проект уже можно запускать из исходников или тестировать через release bundle, но модельные интеграции, packaging и контракты нод продолжают развиваться.

Актуальные архитектурные документы:
- [ARCHITECTURE.md](../ARCHITECTURE.md)
- [docs/NODE_GRAPH_STANDARD.md](NODE_GRAPH_STANDARD.md)
- [docs/node-rules/](node-rules/)

## Требования

- macOS, Linux или Windows в зависимости от backend
- Python 3.11 рекомендуется
- FFmpeg
- CPU/MPS/CUDA в зависимости от выбранного режима исполнения

## Установка

```bash
git clone https://github.com/hockeytek/KeyFlowStudio.git
cd KeyFlowStudio

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Для самого короткого сценария запуска используйте [QUICKSTART.md](../QUICKSTART.md). Подробная установка описана в [docs/installation.md](installation.md).

На macOS FFmpeg можно установить через Homebrew:

```bash
brew install ffmpeg
```

## Запуск приложения

```bash
python main.py
```

## Release Bundle

Готовый macOS Intel preview bundle доступен на странице [GitHub Releases](https://github.com/hockeytek/KeyFlowStudio/releases). Приложение подписано ad-hoc и не notarized через Apple Developer ID, поэтому macOS может попросить подтвердить первый запуск через Open.

Bundle включает код приложения и основные graph templates. Веса моделей и checkpoints в архив не включаются; они настраиваются или загружаются отдельно согласно [models.md](models.md).

## P1 проверка регрессий

```bash
./run_p1.sh
```

Опционально можно указать устройство:

```bash
./run_p1.sh --device cpu
./run_p1.sh --device mps
./run_p1.sh --device auto
```

Скрипт проверяет smoke-кейсы и последовательные прогоны со сменой разрешений в одной сессии модели.

## Структура проекта

```text
KeyFlowStudio/
├── main.py                         # Точка входа и MainWindow
├── requirements.txt                # Python зависимости
├── ARCHITECTURE.md                 # Архитектурный обзор
├── docs/NODE_GRAPH_STANDARD.md     # Стандарт нод и связей
├── app/
│   ├── coordinators/               # Оркестрация runtime и preview
│   ├── node_graph/                 # Specs, rules, engine, properties panels
│   ├── services/                   # Модельные сервисы и inference backend
│   ├── workers/                    # Фоновое исполнение графа
│   └── utils/                      # Медиа, устройство, ffmpeg, диапазоны кадров
├── ec2_worker/                     # Облачный GPU worker
└── tests/                          # Unit, smoke и integration tests
```

## Использование

1. Запустить приложение: `python main.py`.
2. Собрать граф обработки: Source/Load, SAM, BiRefNet, ChromaKey, CorridorKey, MatAnyone2, Write.
3. Настроить свойства нод, пути к медиа, экспорту и устройству выполнения.
4. Запустить выполнение графа.
5. Проверить diagnostics, preview и write output.

## Модели и данные

KeyFlow Studio не распространяет сторонние веса моделей внутри репозитория или release bundle. Проверяйте условия upstream-проектов перед загрузкой, использованием или распространением весов и generated assets.

Для вопросов безопасности используйте приватный канал связи с владельцем репозитория. Не публикуйте credentials, приватные медиафайлы или инфраструктурные детали в issues.