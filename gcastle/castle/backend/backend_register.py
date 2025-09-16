# coding = utf-8

import os
import sys
import logging
import importlib
from ..common.consts import LOG_FORMAT

__all__ = ['get_backend_name', 'set_backend']


logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def set_backend(backend='pytorch'):
    logging.info("You can use ``castle.set_backend(backend)`` or "
                 "``os.environ['CASTLE_BACKEND'] = backend`` "
                 "to set the backend(``pytorch`` or ``mindspore``).")
    logging.info(f"You are using ``{backend}`` as the backend.")
    if backend is None or backend.lower() == 'pytorch':
        os.environ['CASTLE_BACKEND'] = 'pytorch'
    elif backend.lower() == 'mindspore':
        os.environ['CASTLE_BACKEND'] = 'mindspore'
    else:
        raise ValueError(f"backend must be pytorch or mindspore.")
    if 'castle.algorithms' in sys.modules:
        importlib.reload(sys.modules['castle.algorithms'])


def get_backend_name() -> str:
    """Fetch backend from environment variable"""

    backend_name = os.getenv('CASTLE_BACKEND')
    if backend_name is None:
        backend_name = 'pytorch'
        set_backend(backend_name)
    elif backend_name not in ['pytorch', 'mindspore']:
        raise TypeError("Please use ``os.environ['CASTLE_BACKEND'] = backend`` or "
                        "``castle.set_backend(backend)`` "
                        "to set backend environment variable to ``pytorch`` or "
                        "``mindspore``.")
    return backend_name
