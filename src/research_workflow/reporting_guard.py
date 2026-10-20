"""Block ordinary W&B writes while trusted analysis code executes in a dry-run kernel.

This is a capability guard, not an adversarial Python/network sandbox. Custom plotting
code is not executed at all by publish-report --dry-run.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

_READ_METHODS = frozenset(
    {
        "runs",
        "run",
        "artifact",
        "logged_artifacts",
        "used_artifacts",
        "history",
        "scan_history",
        "get",
        "get_dataframe",
        "versions",
        "files",
        "file",
        "keys",
        "items",
        "values",
    }
)
_WRITE_METHODS = frozenset(
    {
        "init",
        "log",
        "log_artifact",
        "finish",
        "login",
        "save",
        "sweep",
        "agent",
        "restore",
        "use_artifact",
    }
)


def _deny(*args, **kwargs):
    raise PermissionError("W&B writes are disabled during reporting dry-run")


def _read_only(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {k: _read_only(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_read_only(v) for v in value]
    if isinstance(value, Iterator):
        return (_read_only(v) for v in value)
    # Dataframes are local values returned by read operations, not remote capabilities.
    if type(value).__module__.startswith(("pandas.", "numpy")):
        return value
    return ReadOnly(value)


class ReadOnly:
    def __init__(self, value):
        object.__setattr__(self, "_value", value)

    def __setattr__(self, name, value):
        _deny()

    def __getattr__(self, name):
        if name.startswith("_") or name in {"update", "delete", "save", "log", "log_artifact", "link"}:
            _deny()
        value = getattr(self._value, name)
        if callable(value):
            if name not in _READ_METHODS:
                _deny()
            return lambda *a, **kw: _read_only(value(*a, **kw))
        return _read_only(value)

    def __iter__(self):
        return (_read_only(v) for v in self._value)

    def __len__(self):
        return len(self._value)

    def __getitem__(self, key):
        return _read_only(self._value[key])


def install_read_only_wandb():
    """Install once in the disposable dry-run kernel, before any notebook cell."""
    import wandb

    factory = wandb.Api
    wandb.Api = lambda *a, **kw: ReadOnly(factory(*a, **kw))
    for method in _WRITE_METHODS:
        if hasattr(wandb, method):
            setattr(wandb, method, _deny)
