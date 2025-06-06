# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Union

import numpy as np
import paddle
import paddle.distributed as distributed

from .utils import device_guard

world_size = distributed.get_world_size()


def convert_file_size_to_int(size: Union[int, str]):
    """
    Converts a size expressed as a string with digits an unit (like `"5MB"`) to an integer (in bytes).
    Args:
        size (`int` or `str`): The size to convert. Will be directly returned if an `int`.
    """
    if isinstance(size, int):
        return size
    if size.upper().endswith("GIB"):
        return int(size[:-3]) * (2**30)
    if size.upper().endswith("MIB"):
        return int(size[:-3]) * (2**20)
    if size.upper().endswith("KIB"):
        return int(size[:-3]) * (2**10)
    if size.upper().endswith("GB"):
        int_size = int(size[:-2]) * (10**9)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("MB"):
        int_size = int(size[:-2]) * (10**6)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("KB"):
        int_size = int(size[:-2]) * (10**3)
        return int_size // 8 if size.endswith("b") else int_size
    raise ValueError(
        "`size` is not in a valid format. Use an integer followed by the unit, e.g., '5GB'."
    )


def reduce_tensor(tensor, buffer_size="32MiB"):
    if tensor.dtype == paddle.int8:
        numel = np.prod(tensor.shape)
    else:
        numel = int(paddle.numel(tensor).item())
    buffer_size = convert_file_size_to_int(buffer_size)
    tensor.reshape_([-1])

    send_size = buffer_size // dtype_byte_size(tensor.dtype)

    for x in range(0, numel, send_size):
        part_tensor = tensor[x : min(numel, x + send_size)]
        yield part_tensor, (x, min(numel, x + send_size))


def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.
    """
    if dtype == paddle.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


@paddle.no_grad()
def distributed_gather(tensor: Any, dst: int = 0, group=None, offload=False) -> Any:
    try:
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(
                distributed_gather(t, dst, group, offload) for t in tensor
            )
        if isinstance(tensor, dict):
            return {
                k: distributed_gather(v, dst, group, offload) for k, v in tensor.items()
            }

        output_tensors = None

        is_dst = dst == distributed.get_rank(group=group)
        if is_dst:
            if offload:
                output_tensors = [
                    [] for _ in range(distributed.get_world_size(group=group))
                ]
            else:
                output_tensors = [
                    paddle.empty_like(tensor)
                    for _ in range(distributed.get_world_size(group=group))
                ]
                output_tensors = [
                    t if len(t.shape) > 0 else t[None] for t in output_tensors
                ]

        if offload:
            origin_shape = tensor.shape
            tensor.reshape_([-1])

            for slice_tensor, index in reduce_tensor(tensor):
                slice_output_tensors = None
                if distributed.get_rank(group=group) == dst:
                    slice_output_tensors = [
                        paddle.empty_like(slice_tensor)
                        for _ in range(distributed.get_world_size(group=group))
                    ]
                paddle.distributed.communication.stream.gather(
                    slice_tensor,
                    slice_output_tensors,
                    dst=group.ranks[dst] if group else dst,
                    group=group,
                    sync_op=True,
                    use_calc_stream=False,
                )

                if is_dst:
                    for i in range(len(output_tensors)):
                        output_tensors[i].append(slice_output_tensors[i].cpu().numpy())

            tensor.reshape_(origin_shape)
            if is_dst:
                with device_guard("cpu"):
                    new_output_tensors = []
                    for x in output_tensors:
                        t = np.concatenate(x)
                        t = t.reshape(origin_shape)
                        new_output_tensors.append(t)
                    output_tensors = new_output_tensors

        else:
            paddle.distributed.communication.stream.gather(
                tensor,
                output_tensors,
                dst=group.ranks[dst] if group else dst,
                group=group,
                sync_op=True,
                use_calc_stream=False,
            )

        return output_tensors

    except AssertionError:
        raise AssertionError("Not currently using distributed training")


@paddle.no_grad()
def distributed_allgather(tensor: Any, group=None, offload=False):
    """nested all gather function with offload

    Args:
        tensor (Any): the desired tensor, list of tensor, dict of tensor to allgather.
        group (_type_, optional): the communication group. Defaults to None.
        offload (bool, optional): If True, we offload the received tensor to cpu/(numpy). Defaults to False.

    Raises:
        AssertionError: Unexpected errors.

    Returns:
        tensor list: list of all gathered tensors
    """
    try:
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(
                distributed_allgather(t, group, offload) for t in tensor
            )
        if isinstance(tensor, dict):
            return {
                k: distributed_allgather(v, group, offload) for k, v in tensor.items()
            }

        output_tensors = []

        if offload:
            with device_guard("cpu"):
                output_tensors = [
                    paddle.empty_like(tensor)
                    for _ in range(distributed.get_world_size(group))
                ]
        else:
            output_tensors = [
                paddle.empty_like(tensor)
                for _ in range(distributed.get_world_size(group))
            ]

        output_tensors = [t if len(t.shape) > 0 else t[None] for t in output_tensors]

        if offload:
            origin_shape = tensor.shape
            tensor.reshape_([-1])
            for x in output_tensors:
                x.reshape_([-1])

            for slice_tensor, index in reduce_tensor(tensor):
                slice_output_tensors = [
                    paddle.empty_like(slice_tensor)
                    for _ in range(distributed.get_world_size(group))
                ]
                distributed.all_gather(slice_output_tensors, slice_tensor, group=group)
                for x, y in zip(slice_output_tensors, output_tensors):
                    with device_guard("cpu"):
                        y[index[0] : index[1]] = x.cpu()

            tensor.reshape_(origin_shape)
            for x in output_tensors:
                x.reshape_(origin_shape)

        else:
            distributed.all_gather(output_tensors, tensor)

        return output_tensors

    except AssertionError:
        raise AssertionError("Not currently using distributed training")
