# coding = utf-8

import warnings
import functools


def deprecated_cls(new_name, message=None):
    """This is a decorator which can be used to mark class as deprecated."""

    def deprecated_func(cls):
        _old_name = cls.__name__
        if new_name is None:
            dep_doc = f"{_old_name} is deprecated from current version " \
                      f"and will be removed in a future vision!"
        else:
            dep_doc = f"{_old_name} is deprecated from current version " \
                      f"and will be removed in a future vision, use {new_name} instead!"
        if message is not None:
            dep_doc = '\t' + dep_doc + '\n\t' + message
        else:
            dep_doc = '\t' + dep_doc
        warnings.warn(dep_doc, DeprecationWarning, 4)

        cls.__name__ = _old_name
        doc = cls.__doc__
        if doc is None:
            doc = dep_doc
        else:
            doc = '\n\n'.join([dep_doc, doc])
        cls.__doc__ = doc
        return cls
    return deprecated_func


def deprecated_func(new_name, message=None):
    """This is a decorator which can be used to mark function as deprecated."""

    def deprecated(func):
        _old_name = func.__name__
        if new_name is None:
            dep_doc = f"{_old_name} is deprecated from current version " \
                      f"and will be removed in a future vision!"
        else:
            dep_doc = f"{_old_name} is deprecated from current version " \
                      f"and will be removed in a future vision, use {new_name} instead!"
        if message is not None:
            dep_doc = '\t' + dep_doc + '\n\t' + message
        else:
            dep_doc = '\t' + dep_doc
        warnings.warn(dep_doc, DeprecationWarning, 4)

        func.__name__ = _old_name
        doc = func.__doc__
        if doc is None:
            doc = dep_doc
        else:
            doc = '\n\n'.join([dep_doc, doc])
        func.__doc__ = doc

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return deprecated
