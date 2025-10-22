import importlib
from pathlib import Path

for path_module in Path(__file__).parent.glob("*.py"):
    name_module = path_module.stem
    if name_module != "__init__":
        module = importlib.import_module(f"{__package__}.{name_module}")
        for name, obj in vars(module).items():
            if isinstance(obj, type):
                globals()[name] = obj
