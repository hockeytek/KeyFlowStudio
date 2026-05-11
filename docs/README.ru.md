# KeyFlow Studio

Qt-приложение для node-based видео keying и matting workflow на базе MatAnyone2, SAM, BiRefNet, CorridorKey и связанных сервисов.

Актуальные архитектурные документы:
- [ARCHITECTURE.md](../ARCHITECTURE.md)
- [docs/NODE_GRAPH_STANDARD.md](NODE_GRAPH_STANDARD.md)

## Требования

- macOS 10.13 или выше, Linux или Windows в зависимости от backend
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

На macOS FFmpeg можно установить через Homebrew:

```bash
brew install ffmpeg
```

## Запуск приложения

```bash
python main.py
```

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
4. Нажать Start Processing.
5. Проверить diagnostics, preview и write output.

## Важно для GitHub

Секреты, access keys, модели, checkpoints, `.venv`, локальные настройки VS Code и результаты прогонов не должны попадать в репозиторий. Они игнорируются через `.gitignore`.