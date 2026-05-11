# KeyFlow Studio - Quick Start

Для актуальной схемы нод и правил совместимости смотрите [docs/NODE_GRAPH_STANDARD.md](docs/NODE_GRAPH_STANDARD.md).

## Первый запуск (одна команда)

```bash
cd /Volumes/MAC\ MEDIA/Temp/KeyFlowStudio
bash setup.sh
```

Скрипт автоматически:
1. ✓ Проверит Python и FFmpeg
2. ✓ Создаст виртуальное окружение
3. ✓ Установит все зависимости (PySide6, PyTorch, OpenCV и т.д.)
4. ✓ Попросит указать путь к MatAnyone2 (или загрузить из GitHub)

## После установки

### Быстрый запуск:
```bash
bash run.sh
```

### Или вручную:
```bash
source .venv/bin/activate
python main.py
```

## Использование приложения

1. **Открыть приложение**
   - загрузить исходное медиа
   - открыть node graph editor

2. **Собрать минимальный граф**
   - выбрать встроенный preset графа или использовать starter buttons в пустом node graph editor
   - `Source/Load -> Matting or keying node -> Write`
   - для масочных сценариев можно использовать `SAM`, `BiRefNet` или `ChromaKey`

3. **Запустить обработку**
   - приложение провалидирует граф
   - затем выполнит его в фоне

4. **Проверить результат**
   - итоговые файлы появятся в write-выходах или в стандартной выходной папке
   - diagnostics покажет ошибки связей или обязательных входов

## Требования

- **macOS 10.13+**
- **Python 3.11 (рекомендуется)**
- **FFmpeg** (установится через `brew install ffmpeg`)

## Если MatAnyone2 уже установлена

Если у вас уже есть локальная копия MatAnyone2 с зависимостями, просто добавьте:

```bash
source .venv/bin/activate
pip install -e /путь/к/MatAnyone2
python main.py
```

## Проблемы?

### FFmpeg не установлен
```bash
brew install ffmpeg
```

### Модель не загружается
Убедитесь что интернет доступен - модель загружается при первом запуске.

### MatAnyone2 не найдена
Установите через setup.sh или вручную:
```bash
pip install git+https://github.com/pq-yang/MatAnyone2.git
```
или если у вас есть локальная копия:
```bash
pip install -e /path/to/MatAnyone2
```

---

**Готово! Следуйте инструкциям выше и запускайте обработку видео.**
