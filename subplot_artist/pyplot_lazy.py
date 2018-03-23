import importlib

__all__ = ['pyplot_lazy_import']


class _PyPlotLazy:

    def __init__(self):
        self._pyplot = None

    @property
    def pyplot(self):
        if self._pyplot is None:
            self._pyplot = importlib.import_module('matplotlib.pyplot')
        return self._pyplot


pyplot_lazy_import = _PyPlotLazy()
