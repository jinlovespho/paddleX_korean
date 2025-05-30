// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "ultra_infer/core/fd_tensor.h"

namespace ultra_infer {
namespace function {

/**
 * @brief Performs sorting on the input tensor along the given axis and outputs
 *        two tensors, Output(Out) and Output(Indices). They reserve the same
 *        shape with Input(X), and Output(Out) represents the sorted tensor
 *        while Output(Indices) gives the sorted order along the given axis
 *        Attr(axis).
 * @param  x            The input of sort
 * @param  out          The sorted tensor of sort op, with the same shape as
 *                      x
 * @param  indices      The indices of a tensor giving the sorted order, with
 *                      the same shape as x
 * @param  axis         The axis along which to sort the tensor.
 *                      When axis < 0, the actual axis will be the |axis|'th
 *                      counting backwards
 * @param  descending   The descending attribute is a flag to tell
 *                      algorithm how to sort the input data.
 *                      If descending is true, will sort by descending order,
 *                      else if false, sort by ascending order
 * @param  indices_type The data type of indices, default to int64
 */
ULTRAINFER_DECL void Sort(const FDTensor &x, FDTensor *out, FDTensor *indices,
                          int axis = 0, bool descending = false,
                          FDDataType indices_type = FDDataType::INT64);

} // namespace function
} // namespace ultra_infer
