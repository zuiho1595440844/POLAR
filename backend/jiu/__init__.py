# backend/jiu/dlgo/__init__.py
# 兼容旧代码里的 "from dlgo.xxx import ..." 绝对导入写法

import sys as _sys
import importlib as _importlib
import pkgutil as _pkgutil

# 1) 把当前包注册为顶层包名 "dlgo"
_sys.modules.setdefault("dlgo", _sys.modules[__name__])

# 2) 将本包下所有一层子模块/子包注册为 "dlgo.<name>"
#    这样 "import dlgo.gotypes" 会映射到 backend.jiu.dlgo.gotypes
for _finder, _name, _ispkg in _pkgutil.iter_modules(__path__):
    _full = f"{__name__}.{_name}"
    try:
        _mod = _importlib.import_module(_full)
        _sys.modules[f"dlgo.{_name}"] = _mod
    except Exception:
        # 某些模块可能依赖顺序；失败也不致命，下方再显式兜底
        pass

# 3) 常用模块显式兜底一次（防止某些环境下 iter_modules 时机问题）
try:
    from . import goboard as _goboard
    from . import gotypes as _gotypes
    from . import scoring as _scoring
    from . import utils as _utils
    _sys.modules["dlgo.goboard"] = _goboard
    _sys.modules["dlgo.gotypes"] = _gotypes
    _sys.modules["dlgo.scoring"] = _scoring
    _sys.modules["dlgo.utils"] = _utils
except Exception:
    pass
