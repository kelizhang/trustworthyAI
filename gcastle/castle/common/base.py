# coding=utf-8
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

import os
import abc
import random
import logging
import numpy as np
import pandas as pd


class BaseLearner(metaclass=abc.ABCMeta):

    def __init__(self):

        self._causal_matrix = None

    @abc.abstractmethod
    def learn(self, data, *args, **kwargs):

        raise NotImplementedError

    @property
    def causal_matrix(self):
        return self._causal_matrix

    @causal_matrix.setter
    def causal_matrix(self, value):
        self._causal_matrix = value


class GradientBasedLearner(BaseLearner):
    def __init__(self):
        super(GradientBasedLearner, self).__init__()
        self.backend = os.environ['CASTLE_BACKEND'].lower()
        self.device = None

    def set_context(self, device_type=None, device_id=None, **kwargs):
        """
        Set device for GradientBased algorithm

        Parameters
        ----------
        device_type: str
            The target device to run, support "cpu", "gpu", and "npu".
        device_id: int
            ID of the target device, the value must be in [0, device_num_per_host-1], Default: 0.
            e.g. device_id=0.

        Other Parameters
        ----------------
            See Also:
            mindspore.set_context()
        """
        if self.backend == 'mindspore':
            import mindspore as ms
            ms.set_context(**kwargs)
        elif self.backend == 'pytorch':
            if device_type is None:
                self.device_type = 'cpu'
            else:
                self.device_type = device_type
            if device_id is None:
                self.device_id = 0
            else:
                self.device_id = device_id

            if self.device_type == 'gpu':
                device = self._to_gpu()
            elif self.device_type == 'npu':
                device = self._to_npu()
            else:
                device = self._to_cpu()
            self.device = device
        else:
            pass

    def _to_cpu(self):
        import torch
        device = torch.device('cpu')
        torch.set_default_dtype(torch.double)
        torch.set_default_device(device)
        return device

    def _to_gpu(self):
        import torch
        if torch.cuda.is_available():
            logging.info('GPU is available.')
            device = torch.device(f'cuda:{self.device_id}')
            torch.set_default_dtype(torch.double)
            torch.set_default_device(device)
        else:
            logging.warning('GPU is unavailable. Reset the device target to CPU.')
            self.device_type = 'cpu'
            device = self._to_cpu()

        return device

    def _to_npu(self):
        import torch
        import torch_npu
        import transfer_to_npu
        if torch.npu.is_available():
            logging.info('NPU is available.')
            device = torch.device(f'npu:{self.device_id}')
            torch.set_default_dtype(torch.float)
            torch.set_default_device(device)
        else:
            logging.warning('NPU is unavailable. Reset the device target to CPU.')
            self.device_type = 'cpu'
            device = self._to_cpu()

        return device

    def _set_seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        if self.backend == 'mindspore':
            import mindspore as ms
            ms._set_seed(seed)
        elif self.backend == 'pytorch':
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        else:
            pass
        try:
            os.environ['PYTHONHASHSEED'] = str(seed)
        except Exception:
            pass


class Tensor:
    """
    A class which is implemented based on numpy.ndarray or pandas.DataFrame

    This object has two special attributes `index` and `columns`.
    The `index` indicates the index of the first axis of the data, the `columns` indicates the index
    of the second axis of the data. In general causal discovery tasks, often use the second axis
    indicates variable or node information(i.e. self.shape[1] == the number of variables).

    Parameters
    ----------
    data: array-like
        list, numpy.array and pandas.DataFrame are supported.
    index: List
        It denotes index of the axis[0] of the data, the size must be equal to self.shape[0].
        If None, range(self.shape[0]) is used by default.
    columns: List
        It denotes index of the axis[1] of the data., the size must be equal to self.shape[1].
        If None, range(self.shape[1]) is used by default.

    Attributes
    ----------
    self.T
    self.ndim
    self.shape
    self.size
    self.index
    self.columns
    self.dtype
    :: For other attributes that are contained in numpy.ndarray, you can use `self.data.` instead.

    Examples
    --------
    First, this object support function `print()` and slicing.
    >>> x = [[1, 2, 3], [1, 2, 3], [2, 1, 3], [2, 1, 3]]
    >>> index = ['A', 'B', 'C', 'D']
    >>> columns = ['a', 'b', 'c']
    >>> arr = Tensor(x, index=index, columns=columns)
    >>> print(arr)
    data:
    [[1 2 3]
     [1 2 3]
     [2 1 3]
     [2 1 3]]
    index: ['A', 'B', 'C', 'D'], columns: ['a', 'b', 'c'], dtype: int64
    >>> print(arr[0:3, 0:2])
    data:
    [[1 2]
     [1 2]
     [2 1]]
    index: ['A'], columns: ['a', 'b'], dtype: int64

    Second, you can use this Tensor to operate like `+`, `-`, `*` and `/` or '//'
    >>> arr2 = arr * 2
    >>> print(arr2)
    data:
    [[2 4 6]
     [2 4 6]
     [4 2 6]
     [4 2 6]]
    index: ['A', 'B', 'C', 'D'], columns: ['a', 'b', 'c'], dtype: int64
    >>> arr3 = arr * arr2
    >>> print(arr3)
    data:
    [[ 2  8 18]
     [ 2  8 18]
     [ 8  2 18]
     [ 8  2 18]]
    index: ['A', 'B', 'C', 'D'], columns: ['a', 'b', 'c'], dtype: int64

    Third, you can use .T to get transpose of a Tensor, but just support 2-dimension data, because
    it will exchange index and columns. Here is a sample:
    >>> print(arr3.T)
    data:
    [[ 2  2  8  8]
     [ 8  8  2  2]
     [18 18 18 18]]
    index: ['a', 'b', 'c'], columns: ['A', 'B', 'C', 'D'], dtype: int64

    Notes::
        A better suggestion is that use this class to operate 2D or 3D data.
    """


    def __init__(self, data, index=None, columns=None):
        if isinstance(data, list):
            data = np.array(data)
        if data.ndim < 2:
            raise ValueError(f"The ndim of data must >= 2.")
        if isinstance(data, Tensor):
            self.data = data.data
            self._index = data.index
            self._columns = data.columns
        elif isinstance(data, np.ndarray):
            self.data = data
            if index is not None:
                if len(index) != data.shape[0]:
                    raise ValueError("Size of index must match axis[0] of the data, "
                                     "which means len(values) == self.shape[0] .")
                self._index = index
            else:
                self._index = list(range(data.shape[0]))
            if columns is not None:
                if len(columns) != data.shape[1]:
                    raise ValueError("Size of columns must match axis[1] of the data, "
                                     "which means len(values) == self.shape[1] .")
                self._columns = columns
            else:
                self._columns = [f"x_{i}" for i in range(data.shape[1])]
        elif isinstance(data, pd.DataFrame):
            self.data = data.values
            self._columns = data.columns.tolist()
            self._index = data.index.tolist()
        else:
            raise ValueError("Unsupported data type. [list, numpy.ndarray, pandas.DataFrame] "
                             "are supported.")

    def __str__(self):
        return f"data: \n{self.data} \n" \
               f"index: {self.index}, columns: {self.columns}, dtype: {self.dtype}"

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        index = list(index)
        i = index[0]
        j = index[1]
        if isinstance(i, int):
            i = [i, i + 1]
        if not isinstance(i, slice):
            index[0] = slice(*i)
        if isinstance(j, int):
            j = [j, j + 1]
        if not isinstance(j, slice):
            index[1] = slice(*j)
        index = tuple(index)
        return Tensor(self.data[index], index=self.index[index[0]], columns=self.columns[index[1]])

    def __setitem__(self, key, value):
        self.data[key] = value

    def __add__(self, other):
        new_data = self.__operate(other, opt='+')
        return Tensor(new_data, index=self.index, columns=self.columns)

    def __sub__(self, other):
        new_data= self.__operate(other, opt='-')
        return Tensor(new_data, index=self.index, columns=self.columns)

    def __mul__(self, other):
        new_data = self.__operate(other, opt='*')
        return Tensor(new_data, index=self.index, columns=self.columns)

    def __truediv__(self, other):
        new_data= self.__operate(other, opt='/')
        return Tensor(new_data, index=self.index, columns=self.columns)

    def __floordiv__(self, other):
        new_data= self.__operate(other, opt='//')
        return Tensor(new_data, index=self.index, columns=self.columns)

    def __array__(self):
        return self.data

    def __operate(self, other, opt):
        if isinstance(other, Tensor):
            if (self.index != other.index or self.columns != other.columns):
                logging.warning("The index or columns for Tensor that add operation are not equal, "
                                "just remain self's.")
            else:
                pass
        if isinstance(other, (int, list, np.ndarray)):
            new_data = eval(f"self.data {opt} other")
        elif isinstance(other, Tensor):
            new_data = eval(f"self.data {opt} other.data")
        else:
            raise ValueError(f"Unsupported operand type(s) for {opt}: {type(other)} and Tensor.")
        return new_data

    @property
    def T(self):
        if self.ndim != 2:
            raise ValueError(f"Unsupported .T operation when self.ndim != 2. "
                             f"You can use self.data.T instead, that will return a numpy.ndarray.")
        return Tensor(self.data.T, index=self.columns, columns=self.index)

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, values: list):
        if len(values) != self.shape[1]:
            raise ValueError(f"Size of the values must match axis[1] of self.data, "
                             f"which means len(values) == self.shape[1] .")
        self._columns = values

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, values: list):
        if len(values) != self.shape[0]:
            raise ValueError(f"Size of the values must match axis[0] of self.data, "
                             f"which means len(values) == self.shape[0] .")
        self._index = values

    def to_numpy(self):
        """
        Convert type to numpy.ndarray

        Notice: This operand will lose the nodes information
        """
        return self.data

    def astype(self, dtype):
        self.data = self.data.astype(dtype)

    @classmethod
    def from_numpy(cls, data):
        return cls(data)


if __name__ == '__main__':
    x = [[1, 2, 3], [1, 2, 3], [2, 1, 3], [2, 1, 3]]
    index = ['A', 'B', 'C', 'D']
    columns = ['a', 'b', 'c']
    arr = Tensor(x, index=index, columns=columns)
    print(arr)
    np.max(arr, axis=1)
    print(arr[0, 0:2])  # 切片依然保持索引

    print(arr // [1, 2, 3])  # 可以 + - * / 运算，运算后的索引保持不变
    print(arr / [1, 2, 3])
    print(arr + [1, 2, 3])
    print(arr - [1, 2, 3])
    print(arr * [1, 2, 3])
    print(arr + arr)

    print(arr.T)    # 对于二维Tensor  提供转置操作 因为行列标签要调换 不适用于多维
    print(arr.ndim)
    print(arr.shape)
    print(arr.size)

    arr2 = arr.to_numpy()
    print(arr2)
    # or
    arr3 = arr.data
    print(arr3)

    print(arr)
