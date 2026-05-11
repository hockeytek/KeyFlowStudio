# KeyFlow Studio

Qt-приложение для node-based видео keying и matting workflow на базе MatAnyone2, SAM, BiRefNet, CorridorKey и связанных сервисов.

Актуальные архитектурные документы:
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [docs/NODE_GRAPH_STANDARD.md](docs/NODE_GRAPH_STANDARD.md)

## Требования

- macOS 10.13 или выше
- Python 3.11 (рекомендуется для совместимости на macOS)
- FFmpeg (устанавливается через Homebrew)

## Установка

### 1. Подготовка

```bash
# Установить Homebrew если его нет
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Установить FFmpeg
brew install ffmpeg

# Перейти в папку проекта
cd /Volumes/MAC\ MEDIA/Temp/KeyFlowStudio
```

### 2. Создать виртуальное окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3. Установить зависимости

```bash
# Базовые зависимости
pip install -r requirements.txt

# MatAnyone2 (если у вас есть локальная копия)
# Замените путь на нужный:
pip install -e /путь/к/KeyFlowStudio
```

### 4. Запуск приложения

```bash
python main.py
```

### 5. P1 проверка регрессий (smoke + смена разрешений)

```bash
./run_p1.sh
```

Опционально можно указать устройство:

```bash
./run_p1.sh --device cpu
./run_p1.sh --device mps
./run_p1.sh --device auto
```

Скрипт проверяет:
- smoke-кейс для изображения,
- smoke-кейс для короткой последовательности кадров,
- последовательные прогоны с переключением разрешений в одной сессии модели (чтобы ловить размерные регрессии).

## Структура проекта

```
KeyFlow Studio/
├── main.py                         # Точка входа и MainWindow
├── requirements.txt                # Python зависимости
├── README.md                       # Этот файл
├── ARCHITECTURE.md                 # Архитектурный обзор
├── docs/NODE_GRAPH_STANDARD.md     # Стандарт нод и связей
├── app/
│   ├── coordinators/               # Оркестрация runtime и preview
│   ├── node_graph/                 # Specs, rules, engine, properties panels
│   ├── services/                   # Модельные сервисы и inference backend
│   ├── workers/                    # Фоновое исполнение графа
│   └── utils/                      # Медиа, устройство, ffmpeg, диапазоны кадров
└── tests/                          # Unit, smoke и integration tests
```

## Использование

1. **Запустить приложение**
   ```bash
   python main.py
   ```

2. **Собрать граф обработки**
   - выбрать источник через Source или Load
   - для быстрого старта можно выбрать встроенный graph preset в основном окне или шаблон прямо в пустом node graph editor
   - добавить нужные узлы: SAM, BiRefNet, ChromaKey, CorridorKey, MatAnyone2, Write
   - соединить порты по совместимым типам

3. **Настроить свойства нод**
   - пути к медиа и экспортам
   - параметры keying/matting
   - устройство выполнения при необходимости

4. **Нажать Start Processing**
   - граф валидируется
   - при ошибках открывается diagnostics
   - при успехе строится execution plan и запускается фоновый runtime

5. **Результаты**
   - сохраняются в write-таргеты или стандартный выходной каталог
   - preview и итоговые пути маршрутизируются обратно в UI

## Оптимизация производительности

### На Apple Silicon (M1/M2/M3):
- Автоматически используется Metal Performance Shaders (MPS)
- Скорость близка к GPU на NVIDIA благодаря встроенным искусственным нейросетям

### На Intel Mac:
- Используется CPU (медленнее)
- Для ускорения рекомендуется использовать внешний GPU если доступен

### Работа с большими видео:
- Запуск на фоновом потоке (не зависает UI)
- Кнопка Cancel стоит во время обработки
- В случае нехватки памяти приложение корректно завершится с ошибкой

## Устранение проблем

### FFmpeg не найден
```bash
brew install ffmpeg
```

### Модель не загружается
- Проверьте интернет-соединение (модель загружается из GitHub при первом запуске)
- Убедитесь, что домашняя папка пользователя доступна для записи (модели кэшируются в `~/Library/Application Support/com.keyflow.studio/models` на macOS)
- При необходимости задайте кастомный путь через переменную `KEYFLOW_MODELS_DIR`

### Приложение зависает
- Долгие операции должны идти в фоне; если UI подвисает, это уже повод для проверки
- Следите за логом и diagnostics окнами
- Нажмите Cancel если нужно остановить run

### Ошибки при импорте MatAnyone2
- Убедитесь что установили: `pip install -e /path/to/KeyFlowStudio`
- Проверьте что MatAnyone2 совместима с вашей версией PyTorch

## Контакты

При вопросах по MatAnyone2:
- GitHub: https://github.com/pq-yang/MatAnyone2
- Email: peiqingyang99@outlook.com
