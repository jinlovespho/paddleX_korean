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

/*! \file runtime.h
    \brief A brief file description.

    More details
 */

#pragma once
#include "ultra_infer/core/fd_tensor.h"
#include "ultra_infer/runtime/backends/backend.h"
#include "ultra_infer/runtime/runtime_option.h"
#include "ultra_infer/utils/perf.h"
/** \brief All C++ UltraInfer APIs are defined inside this namespace
 *
 */
namespace ultra_infer {

/*! @brief Runtime object used to inference the loaded model on different
 * devices
 */
struct ULTRAINFER_DECL Runtime {
public:
  /// Initialize a Runtime object with RuntimeOption
  bool Init(const RuntimeOption &_option);

  /** \brief Inference the model by the input data, and write to the output
   *
   * \param[in] input_tensors Notice the FDTensor::name should keep same with
   * the model's input \param[in] output_tensors Inference results \return true
   * if the inference succeeded, otherwise false
   */
  bool Infer(std::vector<FDTensor> &input_tensors,
             std::vector<FDTensor> *output_tensors);

  /** \brief No params inference the model.
   *
   *  the input and output data need to pass through the BindInputTensor and
   * GetOutputTensor interfaces.
   */
  bool Infer();

  /** \brief Get number of inputs
   */
  int NumInputs() { return backend_->NumInputs(); }
  /** \brief Get number of outputs
   */
  int NumOutputs() { return backend_->NumOutputs(); }
  /** \brief Get input information by index
   */
  TensorInfo GetInputInfo(int index);
  /** \brief Get output information by index
   */
  TensorInfo GetOutputInfo(int index);
  /** \brief Get all the input information
   */
  std::vector<TensorInfo> GetInputInfos();
  /** \brief Get all the output information
   */
  std::vector<TensorInfo> GetOutputInfos();
  /** \brief Bind FDTensor by name, no copy and share input memory
   */
  void BindInputTensor(const std::string &name, FDTensor &input);

  /** \brief Bind FDTensor by name, no copy and share output memory.
   *  Please make share the correctness of tensor shape of output.
   */
  void BindOutputTensor(const std::string &name, FDTensor &output);

  /** \brief Get output FDTensor by name, no copy and share backend output
   * memory
   */
  FDTensor *GetOutputTensor(const std::string &name);

  /** \brief Clone new Runtime when multiple instances of the same model are
   * created
   *
   * \param[in] stream CUDA Stream, default param is nullptr
   * \return new Runtime* by this clone
   */
  Runtime *Clone(void *stream = nullptr, int device_id = -1);

  void ReleaseModelMemoryBuffer();

  RuntimeOption option;

  /** \brief Compile TorchScript Module, only for Poros backend
   *
   * \param[in] prewarm_tensors Prewarm datas for compile
   * \return true if compile succeeded, otherwise false
   */
  bool Compile(std::vector<std::vector<FDTensor>> &prewarm_tensors);
  /** \brief Get profile time of Runtime after the profile process is done.
   */
  double GetProfileTime() {
    return backend_->benchmark_result_.time_of_runtime;
  }

private:
  void CreateOrtBackend();
  void CreatePaddleBackend();
  void CreateTrtBackend();
  void CreateOpenVINOBackend();
  void CreateLiteBackend();
  void CreateRKNPU2Backend();
  void CreateHorizonBackend();
  void CreateSophgoNPUBackend();
  void CreatePorosBackend();
  void CreateTVMBackend();
  void CreateOMBackend();
  std::unique_ptr<BaseBackend> backend_;
  std::vector<FDTensor> input_tensors_;
  std::vector<FDTensor> output_tensors_;
};
} // namespace ultra_infer
