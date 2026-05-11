"""Pytest session configuration.

Problem solved by this file
---------------------------
Two test-isolation issues cause ~95 cascading failures in the full suite:

1. **PySide6 stub contamination** (primary cause, ~90 failures)
   ``test_cloud_region_combo.py`` replaces ``sys.modules["PySide6.QtCore"]``
   (and other PySide6 submodules) with ``MagicMock()`` objects at *module
   level* so that ``app.settings_dialog_mixin`` can be imported in a headless
   environment without a running Qt display.  When ``app.workers.inference_worker``
   is imported *after* this stub is active, ``from PySide6.QtCore import
   QObject, …`` returns a MagicMock attribute.  Python then uses ``MagicMock``
   as the metaclass for ``class InferenceWorker(QObject)``, which produces a
   MagicMock *instance* (not a real class) for ``InferenceWorker``.  Any later
   call to ``InferenceWorker.__new__(InferenceWorker)`` then raises::

       TypeError: issubclass() arg 1 must be a class

   Fix: import ``InferenceWorker`` here, *before* any test file is collected.
   Pytest imports ``conftest.py`` before collecting test modules, so PySide6
   is still intact at this point.  The module is cached in ``sys.modules``
   and all subsequent lazy imports return the real class.

2. **torchvision removal by patch.dict** (secondary cause)
   ``test_gvm_cpu_float32.py::GVMServiceLoadModelTests.test_wrapper_installed_on_cpu``
   uses ``patch.dict(sys.modules, {"gvm_core": …})``.  Inside the context,
   ``GVMService.load_model()`` calls ``_patch_upblock_upsample_size()`` which
   imports ``diffusers → torchvision``.  On exit, ``patch.dict`` removes every
   module that was *added* during the context.  If ``torchvision`` was not in
   ``sys.modules`` before the context, it gets removed.  A subsequent import
   of ``torchvision`` (via ``CorridorKeyModule → timm``) re-registers an
   already-registered PyTorch C++ dispatch kernel, causing::

       AttributeError: partially initialized module 'torchvision'
           (most likely due to a circular import)

   Fix: import ``torchvision`` here so it is in ``sys.modules`` *before* the
   ``patch.dict`` context starts.  The saved snapshot includes ``torchvision``
   and it is not removed on exit.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Pre-import InferenceWorker so it is cached before test_cloud_region_combo.py
# replaces sys.modules["PySide6.*"] with MagicMock stubs (see note 1 above).
try:
    from app.workers.inference_worker import InferenceWorker  # noqa: F401
    _ = InferenceWorker
except Exception:
    pass  # venv missing optional deps — tests will skip/fail individually

# Pre-import torchvision so patch.dict in test_gvm_cpu_float32.py cannot
# remove it from sys.modules (see note 2 above).
try:
    import torchvision  # noqa: F401
    _ = torchvision
except Exception:
    pass
