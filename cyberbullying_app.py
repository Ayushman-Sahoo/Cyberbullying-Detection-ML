from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
for _name in ("backend", "backend.main", "model", "model.predict"):
    sys.modules.pop(_name, None)

_backend_file = _root / "backend" / "main.py"
_spec = spec_from_file_location("project8_backend_main", _backend_file)

if _spec is None or _spec.loader is None:
    raise RuntimeError("Could not load backend module.")

_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

app = _module.app


import uvicorn

if __name__ == "__main__":
    print("Server is starting...")
    uvicorn.run(app, host="127.0.0.1", port=8000)