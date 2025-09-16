# coding = utf-8
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging


class ClassType:
    """Const class type"""

    GENERAL = 'general'
    ALGORITHMS = 'algorithms'


class ClassRegister(object):

    def __init__(self, cls_type):
        self._cls_type = cls_type
        self._registry = {}

    def register(self):
        def wrapper(target_cls):
            """
            Register class with wrapper function.

            Parameters
            ----------
            target_cls: class need to register
            """
            cls_name = target_cls.__name__
            if cls_name not in self._registry:
                self._registry[cls_name] = target_cls
            return target_cls
        return wrapper

    def __getitem__(self, key):
        return self._registry[key]

    def __setitem__(self, key, value):
        if key is None:
            key = value.__name__
        if key in self._registry:
            logging.warning(f"Key {key} already in registry {self._cls_type}.")
        self._registry[key] = value

    def __contains__(self, key):
        return key in self._registry

    def keys(self):
        """key"""
        return self._registry.keys()


class ClassFactory:

    def __init__(self):
        raise RuntimeError("Can't instantiate abstract class ClassFactory")

    algorithms = ClassRegister(cls_type=ClassType.ALGORITHMS)
    general = ClassRegister(cls_type=ClassType.GENERAL)
