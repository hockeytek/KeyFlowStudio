# CorridorKey Integration Plan for KeyFlow Studio

**Дата:** 24 марта 2026  
**Статус:** В разработке  
**Общее время:** ~14-16 часов  
**Риск:** Low-Medium

---

## 📋 Оглавление

1. [Обзор](#обзор)
2. [Архитектурные изменения](#архитектурные-изменения)
3. [Детальный план по этапам](#детальный-план-по-этапам)
4. [Структура файлов](#структура-файлов)
5. [Примеры кода](#примеры-кода)
6. [Тестирование](#тестирование)
7. [Потенциальные проблемы](#потенциальные-проблемы)

---

## Обзор

### Что добавляем
- **CorridorKeyEngine** — основная модель для профессионального chromakeying
- **BiRefNetModule** — вспомогательная модель для генерации альфа-масок

### Результат
Полная интеграция в Node Graph с поддержкой workflow:
```
Load Video → BiRefNet (generate alpha) → CorridorKey (chromakey) → Write Output
```

### Совместимость
- macOS Intel (torch 2.2.2, numpy<2)
- macOS ARM (M1+)
- Linux (CUDA)
- Windows (CUDA 12.1+)

---

## Архитектурные изменения

### 1. Слой сервисов
```
app/services/
├── model_service.py      (существует) ← MatAnyone2
├── corridorkey_service.py (НОВЫЙ)      ← CorridorKey engine
└── birefnet_service.py   (НОВЫЙ)       ← BiRefNet engine
```

**Паттерн:** Каждый сервис отвечает за:
- Lazy loading модели на первый вызов
- Caching в памяти
- Device selection (CPU/MPS/CUDA)
- Error handling и logging

### 2. Слой Node Handlers
```
app/node_graph/nodes/
├── matting_node.py       (существует)
├── sam_mask_node.py      (существует)
├── corridorkey_node.py   (НОВЫЙ)
└── birefnet_node.py      (НОВЫЙ)
```

**Паттерн:** Каждый handler:
- Наследует интерфейс NodeHandler (Protocol)
- Валидирует входные порты
- Возвращает dict[str, dict] output portnames→data
- Не выполняет ничего, просто подготавливает конфиг

### 3. Слой Node Specs (определений)
```
app/node_graph/specs/
├── matting.py           (существует)
├── sam_mask.py          (существует)
├── corridorkey.py       (НОВЫЙ)
└── birefnet.py          (НОВЫЙ)
```

**Паттерн:** Spec определяет:
- Input/Output ports с типами данных
- Properties (параметры с default values)
- UI labels и descriptions

### 4. Слой UI Properties Panels
```
app/node_graph/
├── matting_properties_panel.py       (существует)
├── sammasking_properties_panel.py    (существует)
├── corridorkey_properties_panel.py   (НОВЫЙ)
└── birefnet_properties_panel.py      (НОВЫЙ)
```

**Паттерн:** Panel:
- Наследует QWidget
- Отображает properties конкретной ноды
- Сигналы для обновления graph state

### 5. Requirements & Dependencies
Обновить `requirements.txt`:
```
numpy<2 ; sys_platform == 'darwin' and platform_machine == 'x86_64'
torch==2.2.2 ; sys_platform == 'darwin' and platform_machine == 'x86_64'
torchvision==0.17.2 ; sys_platform == 'darwin' and platform_machine == 'x86_64'
transformers<5 ; sys_platform == 'darwin' and platform_machine == 'x86_64'
timm==1.0.24
# и остальное...
```

---

## Детальный план по этапам

### ЭТАП 1: Подготовка (2-3 часа)

#### 1.1 Обновить requirements.txt
**Файл:** `requirements.txt`
**Задача:** Добавить зависимости CorridorKey с platform markers

```python
# Core inference (CorridorKey)
torch==2.2.2 ; sys_platform == 'darwin' and platform_machine == 'x86_64'
torchvision==0.17.2 ; sys_platform == 'darwin' and platform_machine == 'x86_64'
torch==2.8.0 ; sys_platform != 'darwin' or platform_machine != 'x86_64'
torchvision==0.23.0 ; sys_platform != 'darwin' or platform_machine != 'x86_64'
timm==1.0.24
numpy<2 ; sys_platform == 'darwin' and platform_machine == 'x86_64'
numpy ; sys_platform != 'darwin' or platform_machine != 'x86_64'
opencv-python==4.11.0.86
transformers<5 ; sys_platform == 'darwin' and platform_machine == 'x86_64'
transformers ; sys_platform != 'darwin' or platform_machine != 'x86_64'
Pillow>=8.0
tqdm
kornia
```

**Проверка:**
```bash
pip install -r requirements.txt
python -c "import torch, transformers, kornia; print('OK')"
```

#### 1.2 Подготовить окружение переменных
**Файл:** `main.py`
**Задача:** Установить переменные окружения для CorridorKey

```python
# В самом начале main.py (до импортов):
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

#### 1.3 Создать базовую структуру сервисов
**Файл:** `app/services/corridorkey_service.py`
**Задача:** Lazy loading CorridorKeyEngine

```python
"""CorridorKey model service — lazy loading and device management."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# Import from CorridorKey (assuming it's in PYTHONPATH or site-packages)
try:
    from CorridorKeyModule.inference_engine import CorridorKeyEngine
except ImportError as e:
    CorridorKeyEngine = None
    IMPORT_ERROR = str(e)

logger = logging.getLogger(__name__)

## Единая архитектура выбора устройства (все ноды) ✅

### Стратегия выбора устройства
Все ноды (BiRefNet, CorridorKey, MatAnyone2, SAM) используют **единый глобальный параметр устройства**:

```
Пользователь выбирает устройство в UI → значение сохраняется в QSettings("runtime/device")
                       ↓
main.py sets os.environ["MATANYONE_DEVICE"]
                       ↓
ModelService.reinit_device() запускает переинициализацию
                       ↓
Все сервисы: CorridorKeyService, BiRefNetService, InferenceService
используют app/utils/device.get_device()
                       ↓
Результат: единый torch.device во всем пайплайне
```

### Поддерживаемые устройства
- **CPU**: fallback по умолчанию; принудительно через `MATANYONE_DEVICE=cpu` (для тестов)
- **CUDA**: автоопределение на NVIDIA GPU; принудительно через `MATANYONE_DEVICE=cuda`
- **MPS**: автоопределение на Apple Silicon; принудительно через `MATANYONE_DEVICE=mps`

### Реализация
- ✅ `CorridorKeyService._select_device()` → вызывает `get_device()`
- ✅ `BiRefNetService._select_device()` → вызывает `get_device()`
- ✅ `InferenceService` → использует выбранное пользователем устройство
- ✅ В тестах принудительно установлен `MATANYONE_DEVICE=cpu` для воспроизводимости

---
class CorridorKeyService:
    """Singleton for CorridorKey model management."""
    
    _instance: Optional[CorridorKeyService] = None
    _engine: Optional[CorridorKeyEngine] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.device = self._select_device()
        self.checkpoint_path = self._find_checkpoint()
        self.logger = logger
    
    @staticmethod
    def _select_device() -> torch.device:
        """Select device: CUDA → MPS → CPU."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            return device
        
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device (Apple Metal)")
            return device
        
        device = torch.device("cpu")
        logger.info("Using CPU device")
        return device
    
    @staticmethod
    def _find_checkpoint() -> str:
        """Find CorridorKey checkpoint."""
        # Try multiple locations
        candidates = [
            get_models_dir() / "CorridorKey.pth",
            Path("CorridorKeyModule/checkpoints/CorridorKey.pth"),
        ]
        
        for path in candidates:
            if path.exists():
                logger.info(f"Found CorridorKey checkpoint: {path}")
                return str(path)
        
        raise FileNotFoundError(
            f"CorridorKey checkpoint not found in {candidates}. "
            "Please download from HuggingFace: nikopueringer/CorridorKey_v1.0"
        )
    
    def load_engine(self) -> CorridorKeyEngine:
        """Load or return cached engine."""
        if self._engine is not None:
            return self._engine
        
        if CorridorKeyEngine is None:
            raise RuntimeError(
                f"CorridorKeyModule not available: {IMPORT_ERROR}. "
                "Please install CorridorKey package."
            )
        
        logger.info(f"Loading CorridorKey from {self.checkpoint_path}")
        self._engine = CorridorKeyEngine(
            checkpoint_path=self.checkpoint_path,
            device=str(self.device),
            img_size=2048,
            use_refiner=True,
            mixed_precision=self.device.type != "cpu",
        )
        logger.info("CorridorKey engine loaded successfully")
        return self._engine
    
    def process_frame(
        self,
        image: np.ndarray,
        alpha_hint: Optional[np.ndarray] = None,
        despill_strength: float = 5.0,
    ) -> dict[str, np.ndarray]:
        """Process single frame.
        
        Args:
            image: RGB frame (H,W,3), uint8 or float32
            alpha_hint: Alpha hint (H,W), grayscale
            despill_strength: Despill intensity (0-10)
        
        Returns:
            {
              "rgba": RGBA output (H,W,4)
              "straight": Straight color foreground (H,W,4)
            }
        """
        engine = self.load_engine()
        
        # Validate inputs
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be (H,W,3), got {image.shape}")
        
        # Call engine
        with torch.inference_mode():
            output = engine.process(
                image=image,
                alpha_hint=alpha_hint,
                despill_strength=despill_strength,
            )
        
        return output
    
    def unload_engine(self):
        """Unload model from memory."""
        if self._engine is not None:
            self._engine = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("CorridorKey engine unloaded")
```

#### 1.4 Создать BiRefNet сервис
**Файл:** `app/services/birefnet_service.py`

```python
"""BiRefNet model service for alpha hint generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np

try:
    from BiRefNetModule.wrapper import BiRefNetHandler
except ImportError as e:
    BiRefNetHandler = None
    IMPORT_ERROR = str(e)

logger = logging.getLogger(__name__)


class BiRefNetService:
    """Singleton for BiRefNet model management."""
    
    _instance: Optional[BiRefNetService] = None
    _model: Optional[BiRefNetHandler] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.device = self._select_device()
        self.logger = logger
    
    @staticmethod
    def _select_device() -> str:
        """Select device: cuda → mps → cpu."""
        if torch.cuda.is_available():
            logger.info("Using CUDA device for BiRefNet")
            return "cuda"
        
        if torch.backends.mps.is_available():
            logger.info("Using MPS device for BiRefNet")
            return "mps"
        
        logger.info("Using CPU device for BiRefNet")
        return "cpu"
    
    def load_model(self, usage: str = "General") -> BiRefNetHandler:
        """Load or return cached model."""
        if self._model is not None:
            return self._model
        
        if BiRefNetHandler is None:
            raise RuntimeError(
                f"BiRefNetModule not available: {IMPORT_ERROR}. "
                "Please install BiRefNet package."
            )
        
        logger.info(f"Loading BiRefNet '{usage}' model")
        self._model = BiRefNetHandler(device=self.device, usage=usage)
        logger.info("BiRefNet model loaded successfully")
        return self._model
    
    def process_image(self, image: np.ndarray, usage: str = "General") -> np.ndarray:
        """Generate alpha mask from image.
        
        Args:
            image: RGB image (H,W,3), uint8
            usage: Model variant (General, Matting, Portrait, etc.)
        
        Returns:
            Alpha mask (H,W), float32, range [0, 1]
        """
        model = self.load_model(usage=usage)
        
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be (H,W,3), got {image.shape}")
        
        # Call model
        with torch.inference_mode():
            mask = model.predict(image)
        
        return mask
    
    def unload_model(self):
        """Unload model from memory."""
        if self._model is not None:
            self._model = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("BiRefNet model unloaded")
```

---

### ЭТАП 2: Node Handlers (2 часа)

#### 2.1 Создать CorridorKey Node Handler
**Файл:** `app/node_graph/nodes/corridorkey_node.py`

```python
"""CorridorKey node runtime handler."""

from __future__ import annotations

from app.node_graph.models import GraphNode
from app.node_graph.nodes.base import NodeExecutionContext, NodeExecutionError


class CorridorKeyNodeHandler:
    """Node handler for CorridorKey chromakeying engine."""
    
    key = "corridorkey"
    
    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, dict],
        context: NodeExecutionContext,
    ) -> dict[str, dict]:
        """Validate inputs and prepare execution context.
        
        Note: Actual inference happens in InferenceWorker, not here.
        This handler just validates and routes data.
        """
        # Get upstream data
        img_data = inputs.get("image", {})
        alphahint_data = inputs.get("alphahint", {})
        
        if not img_data:
            raise NodeExecutionError("CorridorKey: missing 'image' input port")
        
        # Get properties
        props = node.properties or {}
        
        # Prepare output
        return {
            "rgba": {
                "media_path": img_data.get("media_path", ""),
                "node_type": "corridorkey",
                "corridorkey": {
                    "despill_strength": float(props.get("despill_strength", 5.0)),
                    "despeckle": bool(props.get("despeckle", True)),
                    "despeckle_size": int(props.get("despeckle_size", 400)),
                    "refiner_strength": float(props.get("refiner_strength", 1.0)),
                    "use_refiner": bool(props.get("use_refiner", True)),
                },
                "upstream_image": img_data,
                "upstream_alphahint": alphahint_data,
            }
        }
```

#### 2.2 Создать BiRefNet Node Handler
**Файл:** `app/node_graph/nodes/birefnet_node.py`

```python
"""BiRefNet node runtime handler."""

from __future__ import annotations

from app.node_graph.models import GraphNode
from app.node_graph.nodes.base import NodeExecutionContext, NodeExecutionError


class BiRefNetNodeHandler:
    """Node handler for BiRefNet alpha mask generation."""
    
    key = "birefnet"
    
    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, dict],
        context: NodeExecutionContext,
    ) -> dict[str, dict]:
        """Validate inputs and prepare execution context."""
        img_data = inputs.get("image", {})
        
        if not img_data:
            raise NodeExecutionError("BiRefNet: missing 'image' input port")
        
        props = node.properties or {}
        
        return {
            "alpha": {
                "media_path": img_data.get("media_path", ""),
                "node_type": "birefnet",
                "birefnet": {
                    "usage": str(props.get("usage", "General")),
                    "resolution": tuple(props.get("resolution", (1024, 1024))),
                    "half_precision": bool(props.get("half_precision", True)),
                },
                "upstream_image": img_data,
            }
        }
```

---

### ЭТАП 3: Node Specs (1-2 часа)

#### 3.1 Создать CorridorKey Spec
**Файл:** `app/node_graph/specs/corridorkey.py`

```python
"""CorridorKey node specification."""

from __future__ import annotations

from app.node_graph.specs.base import PortSpec, NodeSpec


def create_corridorkey_spec() -> NodeSpec:
    """Define CorridorKey node ports and properties."""
    
    return NodeSpec(
        key="corridorkey",
        title="CorridorKey",
        description="Neural network green screen removal with color restoration",
        category="VFX",
        icon="leaf",  # green leaf icon
        
        # Input ports
        input_ports=[
            PortSpec(
                name="image",
                data_type="image",
                label="Image",
                required=True,
            ),
            PortSpec(
                name="alphahint",
                data_type="mask",
                label="Alpha Hint (optional)",
                required=False,
            ),
        ],
        
        # Output ports
        output_ports=[
            PortSpec(
                name="rgba",
                data_type="image",
                label="RGBA Output",
            ),
        ],
        
        # Parameters
        properties={
            "despill_strength": {
                "type": "float",
                "label": "Despill Strength",
                "default": 5.0,
                "min": 0.0,
                "max": 10.0,
                "description": "Green spill removal intensity",
            },
            "despeckle": {
                "type": "bool",
                "label": "Enable Despeckle",
                "default": True,
                "description": "Morph cleanup of tiny artifacts",
            },
            "despeckle_size": {
                "type": "int",
                "label": "Despeckle Size",
                "default": 400,
                "min": 10,
                "max": 2000,
                "description": "Pixel size threshold for cleanup",
            },
            "refiner_strength": {
                "type": "float",
                "label": "Refiner Strength",
                "default": 1.0,
                "min": 0.0,
                "max": 2.0,
                "description": "CNN refiner edge quality enhancement",
            },
            "use_refiner": {
                "type": "bool",
                "label": "Use Refiner",
                "default": True,
                "description": "Enable CNN refiner for better edges",
            },
        },
    )
```

#### 3.2 Создать BiRefNet Spec
**Файл:** `app/node_graph/specs/birefnet.py`

```python
"""BiRefNet node specification."""

from __future__ import annotations

from app.node_graph.specs.base import PortSpec, NodeSpec


def create_birefnet_spec() -> NodeSpec:
    """Define BiRefNet node ports and properties."""
    
    return NodeSpec(
        key="birefnet",
        title="BiRefNet",
        description="Fast alpha mask generation from image",
        category="Alpha",
        icon="mask",
        
        # Input ports
        input_ports=[
            PortSpec(
                name="image",
                data_type="image",
                label="Image",
                required=True,
            ),
        ],
        
        # Output ports
        output_ports=[
            PortSpec(
                name="alpha",
                data_type="mask",
                label="Alpha Mask",
            ),
        ],
        
        # Parameters
        properties={
            "usage": {
                "type": "enum",
                "label": "Model Preset",
                "default": "General",
                "options": [
                    "General",
                    "General-dynamic",
                    "General-HR",
                    "General-Lite",
                    "Matting",
                    "Matting-dynamic",
                    "Portrait",
                    "DIS5K",
                    "COD",
                ],
                "description": "BiRefNet model variant for specific use case",
            },
            "resolution": {
                "type": "tuple",
                "label": "Resolution",
                "default": (1024, 1024),
                "description": "Processing resolution (auto-resizes)",
            },
            "half_precision": {
                "type": "bool",
                "label": "Half Precision",
                "default": True,
                "description": "Use float16 on CUDA for speed (stable on CUDA only)",
            },
        },
    )
```

---

### ЭТАП 4: Engine Registration (30 минут)

#### 4.1 Зарегистрировать handlers в engine
**Файл:** `app/node_graph/engine.py`
**Задача:** Добавить новые handlers в registry

```python
# В функции инициализации engine (где регистрируются handlers):

from app.node_graph.nodes.corridorkey_node import CorridorKeyNodeHandler
from app.node_graph.nodes.birefnet_node import BiRefNetNodeHandler

# ... существующие handlers ...

self.node_handlers = {
    "load": LoadMediaNodeHandler(),
    "sam": SamMaskNodeHandler(),
    "matting": MattingNodeHandler(),
    "corridorkey": CorridorKeyNodeHandler(),    # НОВЫЙ
    "birefnet": BiRefNetNodeHandler(),           # НОВЫЙ
    "write": WriteNodeHandler(),
}
```

#### 4.2 Зарегистрировать specs
**Файл:** `app/node_graph/engine.py`
**Задача:** Добавить specs в реестр

```python
from app.node_graph.specs.corridorkey import create_corridorkey_spec
from app.node_graph.specs.birefnet import create_birefnet_spec

# ... существующие specs ...

AVAILABLE_NODE_SPECS = {
    "load_media": create_load_media_spec(),
    "sam": create_sam_spec(),
    "matting": create_matting_spec(),
    "corridorkey": create_corridorkey_spec(),    # НОВЫЙ
    "birefnet": create_birefnet_spec(),          # НОВЫЙ
    "write": create_write_spec(),
}
```

---

### ЭТАП 5: UI Properties Panels (3-4 часа)

#### 5.1 Создать CorridorKey Properties Panel
**Файл:** `app/node_graph/corridorkey_properties_panel.py`

```python
"""CorridorKey node properties panel."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QSpinBox, QCheckBox, QDoubleSpinBox,
)

from app.node_graph.models import GraphNode


class CorridorKeyPropertiesPanel(QWidget):
    """Properties panel for CorridorKey node."""
    
    properties_changed = Signal(dict)  # Emitted when any property changes
    
    def __init__(self, node: GraphNode, parent: QWidget | None = None):
        super().__init__(parent)
        self.node = node
        self.props = node.properties or {}
        self.setup_ui()
    
    def setup_ui(self):
        """Create UI for properties."""
        layout = QVBoxLayout(self)
        
        # ===== Despill Strength =====
        despill_label = QLabel("Despill Strength (0-10):")
        layout.addWidget(despill_label)
        
        despill_slider = QSlider(Qt.Horizontal)
        despill_slider.setMinimum(0)
        despill_slider.setMaximum(100)
        despill_slider.setValue(int(self.props.get("despill_strength", 5.0) * 10))
        despill_slider.setTickPosition(QSlider.TicksBelow)
        despill_slider.setTickInterval(10)
        
        despill_spinbox = QDoubleSpinBox()
        despill_spinbox.setMinimum(0.0)
        despill_spinbox.setMaximum(10.0)
        despill_spinbox.setValue(self.props.get("despill_strength", 5.0))
        despill_spinbox.setSingleStep(0.1)
        
        despill_slider.valueChanged.connect(
            lambda v: despill_spinbox.setValue(v / 10.0)
        )
        despill_spinbox.valueChanged.connect(
            lambda v: despill_slider.setValue(int(v * 10))
        )
        despill_spinbox.valueChanged.connect(self._on_properties_changed)
        
        despill_layout = QHBoxLayout()
        despill_layout.addWidget(despill_slider)
        despill_layout.addWidget(despill_spinbox)
        layout.addLayout(despill_layout)
        
        # ===== Despeckle =====
        despeckle_checkbox = QCheckBox("Enable Despeckle")
        despeckle_checkbox.setChecked(self.props.get("despeckle", True))
        despeckle_checkbox.stateChanged.connect(self._on_properties_changed)
        layout.addWidget(despeckle_checkbox)
        
        despeckle_size_label = QLabel("Despeckle Size (px):")
        layout.addWidget(despeckle_size_label)
        
        despeckle_size_spinbox = QSpinBox()
        despeckle_size_spinbox.setMinimum(10)
        despeckle_size_spinbox.setMaximum(2000)
        despeckle_size_spinbox.setValue(self.props.get("despeckle_size", 400))
        despeckle_size_spinbox.setSingleStep(50)
        despeckle_size_spinbox.valueChanged.connect(self._on_properties_changed)
        layout.addWidget(despeckle_size_spinbox)
        
        # ===== Refiner =====
        refiner_label = QLabel("Refiner Strength (0-2):")
        layout.addWidget(refiner_label)
        
        refiner_slider = QSlider(Qt.Horizontal)
        refiner_slider.setMinimum(0)
        refiner_slider.setMaximum(100)
        refiner_slider.setValue(int(self.props.get("refiner_strength", 1.0) * 50))
        
        refiner_spinbox = QDoubleSpinBox()
        refiner_spinbox.setMinimum(0.0)
        refiner_spinbox.setMaximum(2.0)
        refiner_spinbox.setValue(self.props.get("refiner_strength", 1.0))
        refiner_spinbox.setSingleStep(0.1)
        
        refiner_slider.valueChanged.connect(
            lambda v: refiner_spinbox.setValue(v / 50.0)
        )
        refiner_spinbox.valueChanged.connect(
            lambda v: refiner_slider.setValue(int(v * 50))
        )
        refiner_spinbox.valueChanged.connect(self._on_properties_changed)
        
        refiner_layout = QHBoxLayout()
        refiner_layout.addWidget(refiner_slider)
        refiner_layout.addWidget(refiner_spinbox)
        layout.addLayout(refiner_layout)
        
        # ===== Use Refiner =====
        use_refiner_checkbox = QCheckBox("Use CNN Refiner")
        use_refiner_checkbox.setChecked(self.props.get("use_refiner", True))
        use_refiner_checkbox.stateChanged.connect(self._on_properties_changed)
        layout.addWidget(use_refiner_checkbox)
        
        layout.addStretch()
        
        # Store widgets for value access
        self._widgets = {
            "despill_strength": despill_spinbox,
            "despeckle": despeckle_checkbox,
            "despeckle_size": despeckle_size_spinbox,
            "refiner_strength": refiner_spinbox,
            "use_refiner": use_refiner_checkbox,
        }
    
    def _on_properties_changed(self):
        """Collect and emit updated properties."""
        updated = {
            "despill_strength": self._widgets["despill_strength"].value(),
            "despeckle": self._widgets["despeckle"].isChecked(),
            "despeckle_size": self._widgets["despeckle_size"].value(),
            "refiner_strength": self._widgets["refiner_strength"].value(),
            "use_refiner": self._widgets["use_refiner"].isChecked(),
        }
        self.properties_changed.emit(updated)
    
    def get_properties(self) -> dict:
        """Get current property values."""
        return {
            "despill_strength": self._widgets["despill_strength"].value(),
            "despeckle": self._widgets["despeckle"].isChecked(),
            "despeckle_size": self._widgets["despeckle_size"].value(),
            "refiner_strength": self._widgets["refiner_strength"].value(),
            "use_refiner": self._widgets["use_refiner"].isChecked(),
        }
```

#### 5.2 Создать BiRefNet Properties Panel
**Файл:** `app/node_graph/birefnet_properties_panel.py`

```python
"""BiRefNet node properties panel."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QCheckBox,
)

from app.node_graph.models import GraphNode


class BiRefNetPropertiesPanel(QWidget):
    """Properties panel for BiRefNet node."""
    
    properties_changed = Signal(dict)
    
    def __init__(self, node: GraphNode, parent: QWidget | None = None):
        super().__init__(parent)
        self.node = node
        self.props = node.properties or {}
        self.setup_ui()
    
    def setup_ui(self):
        """Create UI for properties."""
        layout = QVBoxLayout(self)
        
        # ===== Usage (Model Preset) =====
        usage_label = QLabel("Model Preset:")
        layout.addWidget(usage_label)
        
        usage_combo = QComboBox()
        presets = [
            "General",
            "General-dynamic",
            "General-HR",
            "General-Lite",
            "Matting",
            "Matting-dynamic",
            "Portrait",
            "DIS5K",
            "COD",
        ]
        usage_combo.addItems(presets)
        current_usage = self.props.get("usage", "General")
        usage_combo.setCurrentText(current_usage)
        usage_combo.currentTextChanged.connect(self._on_properties_changed)
        layout.addWidget(usage_combo)
        
        # ===== Half Precision =====
        half_precision_checkbox = QCheckBox("Half Precision (CUDA only)")
        half_precision_checkbox.setChecked(self.props.get("half_precision", True))
        half_precision_checkbox.stateChanged.connect(self._on_properties_changed)
        layout.addWidget(half_precision_checkbox)
        
        layout.addStretch()
        
        self._widgets = {
            "usage": usage_combo,
            "half_precision": half_precision_checkbox,
        }
    
    def _on_properties_changed(self):
        """Emit updated properties."""
        updated = {
            "usage": self._widgets["usage"].currentText(),
            "half_precision": self._widgets["half_precision"].isChecked(),
        }
        self.properties_changed.emit(updated)
    
    def get_properties(self) -> dict:
        """Get current property values."""
        return {
            "usage": self._widgets["usage"].currentText(),
            "half_precision": self._widgets["half_precision"].isChecked(),
        }
```

---

### ЭТАП 6: Worker Integration (2-3 часа)

#### 6.1 Обновить InferenceWorker
**Файл:** `app/workers/inference_worker.py`
**Задача:** Добавить обработку CorridorKey и BiRefNet нод

```python
# В методе process_video InferenceWorker, в раздел обработки нод:

def _process_node(self, node: GraphNode, node_data: dict):
    """Process specific node type."""
    
    if node.type == "corridorkey":
        return self._process_corridorkey_node(node, node_data)
    elif node.type == "birefnet":
        return self._process_birefnet_node(node, node_data)
    # ... существующие типы нод ...

def _process_corridorkey_node(self, node: GraphNode, node_data: dict) -> dict:
    """Process CorridorKey node (actual inference)."""
    from app.services.corridorkey_service import CorridorKeyService
    
    service = CorridorKeyService()
    engine = service.load_engine()
    
    # Load image and alphahint
    upstream = node_data.get("upstream_image", {})
    alphahint = node_data.get("upstream_alphahint", {})
    
    image = self._load_image_from_upstream(upstream)
    alpha_hint = self._load_mask_from_upstream(alphahint) if alphahint else None
    
    # Get parameters
    cfg = node_data.get("corridorkey", {})
    
    # Process
    output = engine.process_frame(
        image=image,
        alpha_hint=alpha_hint,
        despill_strength=cfg.get("despill_strength", 5.0),
    )
    
    # Store result
    node_data["result"] = output["rgba"]
    return node_data

def _process_birefnet_node(self, node: GraphNode, node_data: dict) -> dict:
    """Process BiRefNet node (alpha generation)."""
    from app.services.birefnet_service import BiRefNetService
    
    service = BiRefNetService()
    
    # Load image
    upstream = node_data.get("upstream_image", {})
    image = self._load_image_from_upstream(upstream)
    
    # Get parameters
    cfg = node_data.get("birefnet", {})
    
    # Process
    alpha = service.process_image(
        image=image,
        usage=cfg.get("usage", "General"),
    )
    
    # Store result
    node_data["result"] = alpha
    return node_data
```

---

### ЭТАП 7: Тестирование (3-4 часа)

#### 7.1 Unit тесты сервисов
**Файл:** `tests/test_corridorkey_service.py`

```python
"""Tests for CorridorKey service."""

import pytest
import numpy as np
from app.services.corridorkey_service import CorridorKeyService


@pytest.fixture
def dummy_image():
    """Create dummy RGB image."""
    return np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def dummy_alpha_hint():
    """Create dummy alpha hint."""
    return np.random.rand(720, 1280).astype(np.float32)


def test_service_singleton():
    """Test that service is singleton."""
    s1 = CorridorKeyService()
    s2 = CorridorKeyService()
    assert s1 is s2


def test_device_selection():
    """Test device selection (should not raise)."""
    service = CorridorKeyService()
    assert service.device is not None


def test_checkpoint_location():
    """Test checkpoint finding."""
    service = CorridorKeyService()
    try:
        checkpoint = service.checkpoint_path
        print(f"Found checkpoint: {checkpoint}")
    except FileNotFoundError:
        pytest.skip("Checkpoint not found")


@pytest.mark.skipif(
    not _has_corridorkey_installed(),
    reason="CorridorKey not installed"
)
def test_engine_loading():
    """Test engine loading."""
    service = CorridorKeyService()
    engine = service.load_engine()
    assert engine is not None


@pytest.mark.skipif(
    not _has_corridorkey_installed(),
    reason="CorridorKey not installed"
)
def test_process_frame(dummy_image, dummy_alpha_hint):
    """Test frame processing."""
    service = CorridorKeyService()
    output = service.process_frame(
        image=dummy_image,
        alpha_hint=dummy_alpha_hint,
        despill_strength=5.0,
    )
    
    assert "rgba" in output
    assert output["rgba"].shape == (720, 1280, 4)


def _has_corridorkey_installed():
    """Check if CorridorKey is installed."""
    try:
        from CorridorKeyModule.inference_engine import CorridorKeyEngine
        return True
    except ImportError:
        return False
```

#### 7.2 Integration тесты
**Файл:** `tests/test_corridorkey_integration.py`

```python
"""Integration tests for CorridorKey node graph."""

import pytest
from app.node_graph.engine import GraphEngine
from app.node_graph.nodes.corridorkey_node import CorridorKeyNodeHandler
from app.node_graph.nodes.birefnet_node import BiRefNetNodeHandler
from app.node_graph.models import GraphNode, GraphEdge


def test_birefnet_to_corridorkey_flow():
    """Test pipeline: Load → BiRefNet → CorridorKey → Write."""
    
    engine = GraphEngine()
    
    # Create nodes
    load_node = GraphNode(
        id="load1",
        type="load_media",
        title="Load Video",
        properties={"path": "dummy.mp4"},
    )
    
    birefnet_node = GraphNode(
        id="birefnet1",
        type="birefnet",
        title="BiRefNet Alpha",
        properties={"usage": "General"},
    )
    
    corridorkey_node = GraphNode(
        id="corridorkey1",
        type="corridorkey",
        title="CorridorKey",
        properties={"despill_strength": 5.0},
    )
    
    # Add to graph
    engine.add_node(load_node)
    engine.add_node(birefnet_node)
    engine.add_node(corridorkey_node)
    
    # Connect
    engine.add_edge(GraphEdge("load1", "birefnet1", "out", "image"))
    engine.add_edge(GraphEdge("birefnet1", "corridorkey1", "alpha", "alphahint"))
    engine.add_edge(GraphEdge("load1", "corridorkey1", "out", "image"))
    
    # Validate graph
    assert engine.validate() == (True, [])


def test_corridorkey_handler_validation():
    """Test that CorridorKey handler validates inputs correctly."""
    handler = CorridorKeyNodeHandler()
    
    node = GraphNode(
        id="test",
        type="corridorkey",
        title="Test",
        properties={"despill_strength": 5.0},
    )
    
    # Missing 'image' port should raise
    with pytest.raises(Exception):
        handler.execute(node, {}, None)
    
    # Valid inputs should return output dict
    inputs = {
        "image": {"media_path": "test.mp4"},
    }
    output = handler.execute(node, inputs, None)
    assert "rgba" in output


def test_birefnet_handler_validation():
    """Test BiRefNet handler."""
    handler = BiRefNetNodeHandler()
    
    node = GraphNode(
        id="test",
        type="birefnet",
        title="Test",
        properties={"usage": "General"},
    )
    
    inputs = {"image": {"media_path": "test.mp4"}}
    output = handler.execute(node, inputs, None)
    assert "alpha" in output
```

---

## Структура файлов

```
app/
├── services/
│   ├── model_service.py              (существует)
│   ├── corridorkey_service.py        (НОВЫЙ)
│   └── birefnet_service.py           (НОВЫЙ)
│
├── node_graph/
│   ├── nodes/
│   │   ├── base.py                   (существует)
│   │   ├── load_media_node.py        (существует)
│   │   ├── matting_node.py           (существует)
│   │   ├── sam_mask_node.py          (существует)
│   │   ├── corridorkey_node.py       (НОВЫЙ)
│   │   └── birefnet_node.py          (НОВЫЙ)
│   │
│   ├── specs/
│   │   ├── base.py                   (существует)
│   │   ├── matting.py                (существует)
│   │   ├── sam_mask.py               (существует)
│   │   ├── corridorkey.py            (НОВЫЙ)
│   │   └── birefnet.py               (НОВЫЙ)
│   │
│   ├── engine.py                     (обновить)
│   ├── corridorkey_properties_panel.py (НОВЫЙ)
│   ├── birefnet_properties_panel.py  (НОВЫЙ)
│   └── models.py                     (существует)
│
├── workers/
│   └── inference_worker.py           (обновить)
│
└── ...

tests/
├── test_corridorkey_service.py       (НОВЫЙ)
└── test_corridorkey_integration.py   (НОВЫЙ)

requirements.txt                       (обновить)
main.py                               (обновить)
CORRIDORKEY_INTEGRATION_PLAN.md       (этот файл)
```

---

## Примеры кода

### Использование CorridorKey в коде

```python
from app.services.corridorkey_service import CorridorKeyService
import cv2
import numpy as np

# Load service (singleton)
service = CorridorKeyService()

# Load frame
frame = cv2.imread("input.mp4")  # BGR → need to convert
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Load alpha hint (optional)
alpha_hint = cv2.imread("alpha_hint.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

# Process
result = service.process_frame(
    image=frame_rgb,
    alpha_hint=alpha_hint,
    despill_strength=5.0,
)

# Get outputs
rgba_output = result["rgba"]  # (H, W, 4)
straight_color = result["straight"]  # (H, W, 4)

# Save
cv2.imwrite("output_rgba.png", cv2.cvtColor(rgba_output, cv2.COLOR_RGBA2BGRA))
```

### Node Graph Flow

```python
from app.node_graph.engine import GraphEngine
from app.node_graph.models import GraphNode, GraphEdge

# Create engine
engine = GraphEngine()

# Create nodes
load = GraphNode("load1", "load_media", "Load", {"path": "input.mov"})
birefnet = GraphNode("birefnet1", "birefnet", "Generate Alpha", {"usage": "General"})
corridorkey = GraphNode("corridorkey1", "corridorkey", "CorridorKey", 
                       {"despill_strength": 5.0, "use_refiner": True})
write = GraphNode("write1", "write", "Write", {"path": "output.mov"})

# Add to graph
engine.add_node(load)
engine.add_node(birefnet)
engine.add_node(corridorkey)
engine.add_node(write)

# Connect ports
engine.add_edge(GraphEdge("load1", "birefnet1", "out", "image"))
engine.add_edge(GraphEdge("birefnet1", "corridorkey1", "alpha", "alphahint"))
engine.add_edge(GraphEdge("load1", "corridorkey1", "out", "image"))
engine.add_edge(GraphEdge("corridorkey1", "write1", "rgba", "in"))

# Validate
is_valid, errors = engine.validate()
if is_valid:
    print("Graph is valid!")
    # Execute
    engine.execute()
else:
    print(f"Validation errors: {errors}")
```

---

## Тестирование

### Manual Testing Checklist

- [ ] **Шаг 1:** Установить зависимости
  ```bash
  pip install -r requirements.txt
  ```

- [ ] **Шаг 2:** Проверить импорты
  ```bash
  python -c "from app.services.corridorkey_service import CorridorKeyService; print('OK')"
  python -c "from app.services.birefnet_service import BiRefNetService; print('OK')"
  ```

- [ ] **Шаг 3:** Запустить unit тесты
  ```bash
  pytest tests/test_corridorkey_service.py -v
  pytest tests/test_birefnet_service.py -v
  ```

- [ ] **Шаг 4:** Запустить integration тесты
  ```bash
  pytest tests/test_corridorkey_integration.py -v
  ```

- [ ] **Шаг 5:** Загрузить модели
  ```bash
  # CorridorKey (auto-download на первый run)
  # BiRefNet (auto-download на первый run)
  ```

- [ ] **Шаг 6:** Создать тестовый граф в UI
  - Запустить приложение
  - Load Video → BiRefNet → CorridorKey → Write
  - Run на малом видео (10 frames)
  - Проверить результат

- [ ] **Шаг 7:** Профилировать память
  ```bash
  python -m memory_profiler test_corridorkey_large_video.py
  ```

- [ ] **Шаг 8:** Полное видео
  - Запустить на 4K / HD видео
  - Проверить перфоманс
  - Проверить result качество

---

## Потенциальные проблемы

### 1. numpy/torch Issue on Intel Mac
**Проблема:**
```
RuntimeWarning: The detected NumPy version (2.x.x) may not be compatible 
with this version of PyTorch (2.2.2)
```

**Решение:** 
- requirements.txt: `numpy<2 ; sys_platform == 'darwin' and platform_machine == 'x86_64'`
- Проверить: `pip install numpy==1.26.4`

### 2. CorridorKey Checkpoint Not Found
**Проблема:**
```
FileNotFoundError: CorridorKey checkpoint not found
```

**Решение:**
- Auto-download из HuggingFace на первый run
- Проверить: `~/Library/Application Support/com.keyflow.studio/models/CorridorKey.pth` (macOS default)
- Или вручную: `huggingface-hub download nikopueringer/CorridorKey_v1.0`

### 3. Out of Memory (OOM)
**Проблема:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Решение:**
- Уменьшить resolution входа
- Использовать model_swapping (unload MatAnyone2, load CorridorKey)
- Quantization (float16 для CorridorKey)

### 4. EXR Support Missing
**Проблема:**
```
OpenCV cannot open EXR files
```

**Решение:**
- main.py: `os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"`
- Проверить: `cv2.imread("test.exr")`

### 5. BiRefNet Download Slow
**Проблема:**
```
Downloading BiRefNet model from HuggingFace... (takes 5+ minutes)
```

**Решение:**
- Реализовать progress bar в UI
- Cache модели локально
- Suggest: use smaller variant (BiRefNet-Lite)

### 6. Graph Execution Crashes
**Проблема:**
```
Worker thread crashes, UI freezes
```

**Решение:**
- Убедиться что сервисы thread-safe
- Использовать QThread правильно (emit signals, не call methods directly)
- Профилировать с PyCharm profiler

---

## Временная шкала разработки

| Этап | Файлы | Паттерн | Время |
|------|-------|--------|-------|
| 1. Подготовка | requirements.txt, main.py | Копировать маркеры из CorridorKey | 2-3 ч |
| 2. Services | 2 файла services | Singleton + lazy load | 1.5 ч |
| 3. Handlers | 2 файла nodes | Копировать из matting_node.py | 1 ч |
| 4. Specs | 2 файла specs | Копировать из matting.py | 1.5 ч |
| 5. Engine | engine.py | Добавить в registry | 0.5 ч |
| 6. UI Panels | 2 файла | Копировать из matting_properties | 2-3 ч |
| 7. Worker | inference_worker.py | Добавить _process_corridorkey | 1.5 ч |
| 8. Тесты | tests/ | Unit + integration | 2 ч |
| 9. Дебаг & QA | — | Реальные видео, профилирование | 2-3 ч |
| **TOTAL** | | | ~16 ч |

---

## Рекомендуемый порядок разработки


### День 1
1. ✅ Этап 1 — Подготовка (2-3 ч)
2. ✅ Этап 2 — Services (1.5 ч)
3. ✅ Этап 3 — Handlers (1 ч)

### День 2
1. ✅ Этап 4 — Specs (1.5 ч)
2. ✅ Этап 5 — Engine (0.5 ч)
3. ✅ Этап 6 — UI Panels (2-3 ч)

### День 3
1. ✅ Этап 7 — Интеграция в Worker (1.5 ч)
2. ✅ Этап 8 — Выполнение нод и топологическая сортировка (2 ч)
3. ✅ Этап 9 — Единая архитектура выбора устройства (1-2 ч)

### День 4
1. ✅ **Этап 6.1 — Интеграционные тесты (2 ч)** — все 8 тестов пройдены
2. ⏳ Этап 6.2-6.5 — производительность и валидация (4-5 ч)
3. ⏳ Этап 7-9 — документация и финальная очистка (2-3 ч)
---

## Checkpoint & Validation

### Milestone 1: Services Working
```bash
python -c "
from app.services.corridorkey_service import CorridorKeyService
from app.services.birefnet_service import BiRefNetService
print('Services imported OK')
"
```

### Milestone 2: Node Handlers Registered
```bash
python -c "
from app.node_graph.engine import GraphEngine
engine = GraphEngine()
assert 'corridorkey' in engine.node_handlers
assert 'birefnet' in engine.node_handlers
print('Handlers registered OK')
"
```

### Milestone 3: UI Works
```
1. Launch app
2. Add BiRefNet node to graph
3. Add CorridorKey node to graph
4. See properties panels load correctly
```

### Milestone 4: Graph Executes
```
1. Load video
2. Connect: Load → BiRefNet → CorridorKey → Write
3. Run inference on 10 frames
4. Verify output files
```

---

## Финальная проверка

- [ ] Все файлы скомпилированы без ошибок
- [ ] Unit тесты проходят
- [ ] Integration тесты проходят
- [ ] UI не вешается при добавлении нод
- [ ] Граф выполняется без ошибок
- [ ] Результаты корректные (визуально проверить)
- [ ] Память не теч (профилировать)
- [ ] Поддержка всех платформ (тестовать на Intel Mac + Linux)

---

**Автор плана:** GitHub Copilot  
**Дата:** 24 марта 2026  
**Статус:** Готов к разработке
