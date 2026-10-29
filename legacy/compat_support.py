"""Populate deprecated source-checkout imports without duplicating implementations."""
from importlib import import_module


def export_module(target,namespace):
    module = import_module(target)
    namespace.update({key:value for key,value in vars(module).items() if not key.startswith("__")})
    namespace["__all__"] = [key for key in vars(module) if not key.startswith("_")]
    namespace["__getattr__"] = lambda name:getattr(module,name)
