# KeyFlow Studio - Checklist & Troubleshooting

## ✅ Установка (Step-by-Step)

### Шаг 1: Подготовка
- [ ] macOS 10.13 или выше
- [ ] Python 3.9+ установлен
- [ ] У вас есть IntegratedTerm файловый менеджер (Finder/Terminal)

### Шаг 2: FFmpeg
```bash
# Проверить установлена ли FFmpeg
ffmpeg -version

# Если нет - установить через Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install ffmpeg
```

### Шаг 3: Загрузка проекта и установка

```bash
# Перейти в папку проекта
cd /Volumes/MAC\ MEDIA/Temp/KeyFlowStudio

# Запустить автоматическую установку
bash setup.sh

# Следуйте указаниям в скрипте (выберите вариант установки MatAnyone2)
```

### Шаг 4: Проверка

```bash
# Активировать виртуальное окружение
source .venv/bin/activate

# Проверить что все установилось
python -c "import PySide6, torch, cv2; print('✓ All OK')"

# Проверить выбранное устройство
python -c "from app.utils import get_device_name; print(f'Device: {get_device_name()}')"
```

### Шаг 5: Запуск

```bash
bash run.sh
# или
python main.py
```

## 🚀 Первый запуск приложения

1. **Окно приложения должно открыться** с вкладками Input, Log & Progress
2. **System Info должна покать ваше устройство** (MPS, CPU или CUDA)
3. **Log должен показать версию FFmpeg**

### Если ничего не произошло:

```bash
# Запустить с verbose логированием
python -u main.py 2>&1 | tee app.log

# Посмотреть ошибки в app.log
cat app.log
```

## 🎬 Первая обработка видео

### Подготовить тестовые файлы

1. **Видеофайл:** любое видео mp4/avi/mov
   - На первый раз: 5-10 секунд (30 кадров), 720p
   - Убедиться что это реальное видео с человеком

2. **Маска:** PNG с белым объектом на черном фоне
   - Размер: такой же как первый кадр видео
   - Можно получить через:
     - PhotoShop/GIMP: Magic Wand + Export
     - SAM2 (Segment Anything): https://huggingface.co/spaces/fffiloni/SAM2-Image-Predictor
     - Или нарисовать в Paint

### Пример:
```bash
# Скачать SAM2 результат маски
# Или нарисовать маску вот так:
python3 << 'EOF'
from PIL import Image
import numpy as np

# Создать белый прямоугольник на черном фоне (тест)
img = Image.new('L', (1280, 720), color=0)
pixels = img.load()
for x in range(300, 700):
    for y in range(200, 600):
        pixels[x, y] = 255
img.save("test_mask.png")
print("✓ Created test_mask.png")
EOF
```

### В приложении:
1. Нажать "Browse" в разделе "Video Input"
2. Выбрать видеофайл
3. Нажать "Browse" в разделе "Mask Input"
4. Выбрать маску
5. Нажать "Start Processing"

## 🔧 Troubleshooting

### ❌ "No module named 'PySide6'"
```bash
source .venv/bin/activate
pip install PySide6
```

### ❌ "No module named 'matanyone2'"
Модель не установлена. Два варианта:

**Вариант A: Из GitHub (автоматически)**
```bash
pip install git+https://github.com/pq-yang/MatAnyone2.git
```

**Вариант B: Локально (если у вас есть копия)**
```bash
pip install -e /path/to/MatAnyone2
```

### ❌ "No module named 'sam2'" (Intel macOS)
На Intel macOS это часто связано с тем, что upstream `sam2` требует `torch>=2.5.1`,
а доступные колеса ограничены веткой `torch 2.2.x`.

Для KeyFlow Studio добавлен workaround-скрипт:

```bash
cd /Volumes/MAC\ MEDIA/Temp/KeyFlowStudio
bash scripts/install_sam2_intel_workaround.sh
```

Что делает скрипт:
- клонирует исходники `sam2`,
- ослабляет требование к версии `torch`,
- ставит пакет в текущий `.venv`,
- проверяет импорт и native init в `Sam2Service`.

Важно: это неофициальный обход. После пересоздания `.venv` его нужно запустить снова.

### ❌ "FFmpeg not found"
```bash
# Проверить установлена ли
which ffmpeg

# Установить если нет
brew install ffmpeg

# Проверить что работает
ffmpeg -version
```

### ❌ "No attribute 'mps'" или "MPS not available"
Это нормально на Intel Mac. Приложение будет использовать CPU.
На Apple Silicon это может быть проблема с версией PyTorch:
```bash
# Переустановить torch для macOS
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### ❌ Приложение зависает при нажатии "Start Processing"
**Это нормально.** Модель загружается в фоне (первый запуск 30-60 сек).
Смотрите лог в "Log & Progress" табе, там должны быть сообщения.

Если вообще нет никаких сообщений:
```bash
# Остановить приложение (Ctrl+C)
# Запустить в терминале с логированием
python -u main.py 2>&1 | tail -20
```

### ❌ "CUDA out of memory" или "MPS out of memory"
Видео слишком большое или разрешение слишком высокое.

Решение:
1. Попробуйте уменьшить разрешение видео:
```bash
ffmpeg -i input.mp4 -vf scale=-1:720 output_720p.mp4
```

2. Или обрабатывайте частями (разные временные интервалы)

### ❌ Результирующее видео пустое или черное
Проблема с маской:
- [ ] Маска должна быть черный фон + белый объект
- [ ] Размер маски должен совпадать с видео
- [ ] Маска должна быть в формате PNG или JPG

Проверить маску:
```python
from PIL import Image
import numpy as np

mask = Image.open("your_mask.png")
print(f"Mode: {mask.mode}, Size: {mask.size}")
print(f"Min: {np.array(mask).min()}, Max: {np.array(mask).max()}")
```

### ❌ "Unexpected keyword argument 'objects'"
У вас старая версия MatAnyone2. Обновите:
```bash
pip install --upgrade matanyone2
# или переустановите из GitHub
pip uninstall matanyone2
pip install git+https://github.com/pq-yang/MatAnyone2.git
```

### ❌ Результаты сохраняются в странном месте
Они сохраняются рядом с видеофайлом:
```
/путь/к/видео/
├── input_video.mp4
└── input_video_keyflow_out/
    ├── alpha/
    │   └── ...
    └── fg/
        └── ...
```

### ❌ Хочу отменить процесс во время обработки
Нажмите кнопку "Cancel" в приложении.
Worker проверяет флаг отмены после каждого кадра.

## 📊 Ожидаемая производительность

### На Apple Silicon (M1/M2/M3):
- 1-2 fps для видео 1080p (в зависимости от модели)
- Видео на 10 сек = примерно 30 сек обработки
- Использование памяти: 2-4 GB

### На Intel Mac (CPU):
- 0.1-0.5 fps (намного медленнее)
- Видео на 10 сек = примерно 5-10 минут
- **Рекомендуется:** уменьшить разрешение

## 🔍 Проверка что работает

```bash
# Test 1: Check device
python -c "from app.utils import get_device, get_device_name; \
  device = get_device(); \
  print(f'✓ Device: {get_device_name()}, Type: {device.type}')"

# Test 2: Check FFmpeg
python -c "from app.utils import check_ffmpeg, get_ffmpeg_info; \
  print(f'✓ FFmpeg: {get_ffmpeg_info()}')"

# Test 3: Check MatAnyone2
python -c "from app.services import ModelService; \
  ms = ModelService(); \
  print('✓ MatAnyone2 can be imported')"

# Test 4: Load model (warning: first time ~1 min)
timeout 120 python -c "from app.services import ModelService; \
  ms = ModelService(); \
  ms.load_model(); \
  print('✓ Model loaded successfully')" || echo "Timeout or error"

# Test 5: Run app
python main.py
```

## 📝 Сбор информации для отладки

Если что-то не работает, соберите эту информацию:

```bash
# Сохранить вывод в файл
{
  echo "=== System ==="
  uname -a
  
  echo "=== Python ==="
  python --version
  which python
  
  echo "=== FFmpeg ==="
  ffmpeg -version 2>&1 | head -n 1
  
  echo "=== PyTorch ==="
  python -c "import torch; print(f'PyTorch: {torch.__version__}')"
  python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
  python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
  
  echo "=== MatAnyone2 ==="
  python -c "from matanyone2 import __version__; print(f'MatAnyone2: {__version__}')" 2>&1 || echo "Not installed"
  
  echo "=== App Test ==="
  python -u main.py 2>&1 || echo "Error starting app"
} | tee debug_info.txt

# Отправить это содержимое при запросе помощи
```

## ✅ Финальный чек-лист

При успешной установке должно быть:

- [ ] `python main.py` запускает окно приложения
- [ ] Окно показывает "Device: ..." (MPS, CPU или CUDA)
- [ ] Окно показывает версию FFmpeg
- [ ] Можно выбрать видео и маску через "Browse"
- [ ] Нажатие "Start" не вызывает ошибку
- [ ] Логирование показав сообщение о загрузке модели
- [ ] После обработки видео создается папка с результатами

Если все это работает - приложение готово к использованию! 🎉

---

**Застряли? Проверьте соответствующий раздел выше или соберите debug info и попробуйте переустановить зависимости.**
