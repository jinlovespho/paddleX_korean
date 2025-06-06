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
from __future__ import annotations

import gc
import os
import re
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import paddle
import paddle.nn as nn
from paddle import Tensor
from paddle.distributed.fleet.meta_parallel.parallel_layers import PipelineLayer

try:
    from paddle.distributed.fleet.meta_parallel import LocalSharedLayerDesc
except:
    LocalSharedLayerDesc = None
from paddle.nn import Layer

from ......utils import logging
from ...tokenizer.tokenizer_utils import InitTrackerMeta, adapt_stale_fwd_patch
from ..generation import GenerationConfig, GenerationMixin
from ..utils import (
    CONFIG_NAME,
    LEGACY_CONFIG_NAME,
    PADDLE_WEIGHTS_INDEX_NAME,
    PADDLE_WEIGHTS_NAME,
    PYTORCH_WEIGHTS_INDEX_NAME,
    PYTORCH_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    device_guard,
    resolve_file_path,
)
from .configuration_utils import PretrainedConfig
from .conversion_utils import ConversionMixin
from .utils import (
    ContextManagers,
    fn_args_to_dict,
    get_checkpoint_shard_files,
    is_safetensors_available,
    paddlenlp_load,
    weight_name_suffix,
)

__all__ = [
    "PretrainedModel",
]


def _add_variant(weights_name: str, variant=None) -> str:
    if variant is not None and len(variant) > 0:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name


@contextmanager
def dtype_guard(dtype="float32"):
    origin_dtype = paddle.get_default_dtype()
    paddle.set_default_dtype(dtype)
    try:
        yield
    finally:
        paddle.set_default_dtype(origin_dtype)


_init_weights = True


@contextmanager
def no_init_weights(_enable=True):
    """
    Context manager to globally disable weight initialization to speed up loading large models.

    TODO(Patrick): Delete safety argument `_enable=True` at next major version. .
    """
    global _init_weights
    old_init_weights = _init_weights
    if _enable:
        _init_weights = False
    try:
        yield
    finally:
        _init_weights = old_init_weights


def _split_keys_evenly(keys: list, n: int) -> list:
    """Split a list into n lists with an equal number of elements.

    Args:
        keys (list): the list to be split
        n (int): number of splits

    Returns:
        result: list of lists
    """

    total_len = len(keys)
    base_size = total_len // n
    extra = total_len % n

    result = []
    index = 0
    for _ in range(n):
        part_size = base_size + 1 if extra > 0 else base_size
        extra -= 1
        result.append(keys[index : index + part_size])
        index += part_size

    return result


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike],
    tensor_parallel_split_mapping=None,
    fliter_dict_keys=None,
    device="cpu",
    ckpt_quant_stage="O0",
):
    """
    Reads a PaddlePaddle checkpoint file, returning properly formatted errors if they arise.
    """

    if tensor_parallel_split_mapping is None:
        tensor_parallel_split_mapping = {}

    state_dict = paddlenlp_load(checkpoint_file, map_location="cpu")
    return state_dict


_re_layer_prefix = re.compile(r"\.(\d+)\.")


def _load_state_dict_into_model(model_to_load, state_dict, start_prefix):
    # torch will cast dtype in load_state_dict, but paddle strictly check dtype
    _convert_state_dict_dtype_and_shape(state_dict, model_to_load)

    error_msgs = []

    if len(start_prefix) > 0:
        for key in list(state_dict.keys()):
            if key.startswith(start_prefix):
                state_dict[key.replace(start_prefix, "")] = state_dict.pop(key)

    # TODO: add return status to state_dict
    with warnings.catch_warnings(record=True) as w:
        warnings.resetwarnings()
        # paddlenlp hold  missing_keys , just ignore not found warnings.
        warnings.filterwarnings(
            "ignore", message=r".*is not found in the provided dict.*"
        )
        warnings.filterwarnings("ignore", message=r".*paddle.to_tensor.*")
        model_to_load.set_state_dict(state_dict)
        error_msgs.extend([str(x.message) for x in w])

    del state_dict

    return error_msgs


def _convert_state_dict_dtype_and_shape(state_dict, model_to_load):
    # convert the dtype of state dict
    def is_0d_or_1d(tensor):
        return len(tensor.shape) == 0 or list(tensor.shape) == [1]

    for key, value in model_to_load.state_dict().items():
        if key in list(state_dict.keys()):
            if isinstance(state_dict[key], np.ndarray):
                raise ValueError(
                    "convert_state_dict_dtype expected paddle.Tensor not numpy.ndarray, please convert numpy.ndarray to paddle.Tensor"
                )
            # confirm parameter cast is executed on the same device as model
            # TODO: cast(FP32 -> FP16) has diff on different devices, need to fix it
            if (
                state_dict[key].is_floating_point()
                and state_dict[key].dtype != value.dtype
            ):
                state_dict[key] = paddle.cast(state_dict.pop(key), value.dtype)
            # unified 0d and 1d tensor
            if is_0d_or_1d(value) and is_0d_or_1d(state_dict[key]):
                if list(value.shape) != list(state_dict[key].shape):
                    state_dict[key] = paddle.reshape(state_dict.pop(key), value.shape)


def _load_state_dict_into_meta_model(
    model,
    state_dict,
    loaded_state_dict_keys,  # left for now but could be removed, see below
    start_prefix,
    expected_keys,
    dtype=None,
    is_safetensors=False,
    keep_in_fp32_modules=None,
):
    """
    This is somewhat similar to `_load_state_dict_into_model`, but deals with a model that has some or all of its
    params on a `meta` device. It replaces the model params with the data from the `state_dict`, while moving the
    params back to the normal device, but only for `loaded_state_dict_keys`.

    `start_prefix` is used for models which insert their name into model keys, e.g. `bert` in
    `bert.pooler.dense.weight`

    """
    from paddle.common_ops_import import convert_np_dtype_to_dtype_

    dtype = convert_np_dtype_to_dtype_(dtype)
    error_msgs = []
    model_state_dict = model.state_dict()
    for param_name, param in state_dict.items():
        # First part of the test is always true as loaded_state_dict_keys always contains state_dict keys.
        if param_name not in loaded_state_dict_keys or param_name not in expected_keys:
            continue

        if param_name.startswith(start_prefix):
            param_name = param_name[len(start_prefix) :]

        if param.place != paddle.framework._current_expected_place():
            param = param._copy_to(paddle.framework._current_expected_place(), False)

        # # We convert floating dtypes to the `dtype` passed. We want to keep the buffers/params
        # # in int/uint/bool and not cast them.
        if dtype is not None and paddle.is_floating_point(param):
            if (
                keep_in_fp32_modules is not None
                and any(
                    module_to_keep_in_fp32 in param_name
                    for module_to_keep_in_fp32 in keep_in_fp32_modules
                )
                and (dtype == paddle.float16 or dtype == paddle.bfloat16)
            ):
                param = param.astype(dtype=paddle.float32)
            else:
                param = param.astype(dtype=dtype)

        if dtype is None:
            old_param = model
            splits = param_name.split(".")
            for split in splits:
                old_param = getattr(old_param, split)
                if old_param is None:
                    break

            if old_param is not None:
                param = param.astype(dtype=old_param.dtype)
        with paddle.no_grad():
            model_state_dict[param_name].get_tensor()._share_data_with(
                param.value().get_tensor()
            )
            param.value().get_tensor()._clear()
    return error_msgs


class PretrainedModel(
    Layer, GenerationMixin, ConversionMixin, metaclass=InitTrackerMeta
):
    """
    The base class for all pretrained models. It mainly provides common methods
    for loading (construction and loading) and saving pretrained models. Loading
    and saving also rely on the following class attributes which should be overridden
    by derived classes accordingly:

    - **model_config_file** (str): Represents the file name of model configuration
        for configuration saving and loading in local file system. The value is
        `model_config.json`.
    - **resource_files_names** (dict): Name of local file where the model configuration
        can be saved and loaded locally. Currently, resources only include the model state,
        thus the dict only includes `'model_state'` as key with corresponding
        value `'model_state.pdparams'` for model weights saving and loading.
    - **pretrained_init_configuration** (dict): Provides the model configurations
        of built-in pretrained models (contrasts to models in local file system).
        It has pretrained model names as keys (such as `bert-base-uncased`), and
        the values are dict preserving corresponding configuration for model initialization.
    - **pretrained_resource_files_map** (dict): Provides resource URLs of built-in
        pretrained models (contrasts to models in local file system).
        It has the same key as resource_files_names (that is "model_state"),
        and the corresponding value is a dict with specific model name to model weights URL mapping
        (such as "bert-base-uncased" ->
        "https://bj.bcebos.com/paddlenlp/models/transformers/bert-base-uncased.pdparams").
    - **base_model_prefix** (str): Represents the attribute associated to the
        base model in derived classes of the same architecture adding layers on
        top of the base model. Note: A base model class is pretrained model class
        decorated by `register_base_model`, such as `BertModel`; A derived model
        class is a pretrained model class adding layers on top of the base model,
        and it has a base model as attribute, such as `BertForSequenceClassification`.

    Methods common to models for text generation are defined in `GenerationMixin`
    and also inherited here.

    Besides, metaclass `InitTrackerMeta` is used to create `PretrainedModel`,
    by which subclasses can track arguments for initialization automatically.
    """

    # Deprecated(wj-Mcat): after 2.6.* version
    # save the old-school `LEGACY_CONFIG_NAME`, and will be changed to `CONFIG_NAME` after 2.6.* version
    model_config_file = LEGACY_CONFIG_NAME

    pretrained_init_configuration = {}
    # TODO: more flexible resource handle, namedtuple with fields as:
    # resource_name, saved_file, handle_name_for_load(None for used as __init__
    # arguments), handle_name_for_save
    resource_files_names = {"model_state": PADDLE_WEIGHTS_NAME}
    pretrained_resource_files_map = {}
    base_model_prefix = ""
    main_input_name = "input_ids"
    config_class = None
    _keep_in_fp32_modules = None

    # a list of `re` patterns of `state_dict` keys that should be removed from the list of missing
    # keys we find (keys inside the model but not in the checkpoint) and avoid unnecessary warnings.
    _keys_to_ignore_on_load_missing = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of
    # unexpected keys we find (keys inside the checkpoint but not the model) and avoid unnecessary
    # warnings.
    _keys_to_ignore_on_load_unexpected = None
    # a list of `state_dict` keys to ignore when saving the model (useful for keys that aren't
    # trained, but which are either deterministic or tied variables)
    _keys_to_ignore_on_save = None
    _tied_weights_keys = None

    def __init__(self, *args, **kwargs):
        super(PretrainedModel, self).__init__()

        if not self.constructed_from_pretrained_config():
            return

        # extract config from args
        config = None
        for arg in args:
            if isinstance(arg, PretrainedConfig):
                config = arg
                break
        if config is not None:
            self.config: PretrainedConfig = config
            self.model_config_file = CONFIG_NAME
            self.generation_config = (
                GenerationConfig.from_model_config(self.config)
                if self.can_generate()
                else None
            )
            return

        # extract config from kwargs
        if "config" not in kwargs:
            raise ValueError(
                "PretrainedConfig instance not found in the arguments, you can set it as args or kwargs with config field"
            )

        config = kwargs["config"]
        if not isinstance(config, PretrainedConfig):
            raise TypeError(
                "config parameter should be the instance of PretrainedConfig"
            )

        self.config: PretrainedConfig = kwargs["config"]
        self.generation_config = (
            GenerationConfig.from_model_config(self.config)
            if self.can_generate()
            else None
        )
        self.model_config_file = CONFIG_NAME
        self.warnings_issued = {}

    def _post_init(self, original_init, *args, **kwargs):
        """
        It would be hooked after `__init__` to add a dict including arguments of
        `__init__` as a attribute named `config` of the pretrained model instance.
        """
        if not self.constructed_from_pretrained_config():
            init_dict = fn_args_to_dict(original_init, *((self,) + args), **kwargs)
            self.config = init_dict

        # only execute when it's the base method
        if (
            original_init.__module__ != "paddlenlp.transformers.model_utils"
            and self.__class__.init_weights is PretrainedModel.init_weights
        ):
            self.init_weights()

        # Note:
        # 1. PipelineLayer will create parameters for each layer and
        # call `_synchronize_shared_weights()` to synchronize the shared parameters.
        # 2. When setting the model `state_dict`, `_synchronize_shared_weights` will be called to
        # synchronize the shared parameters.
        # However, `self._init_weights` will re-initialize the parameters without
        # synchronizing the shared parameters. If the following step does not load a checkpoint,
        # the shared parameters will be different.

        if isinstance(self, PipelineLayer):
            self._synchronize_shared_weights()

    def _init_weights(self, layer):
        """
        Initialize the weights. This method should be overridden by derived class.
        """
        pass

    def _initialize_weights(self, layer):
        """
        Initialize the weights if they are not already initialized.
        """
        if getattr(layer, "_is_initialized", False):
            return
        self._init_weights(layer)
        layer._is_initialized = True

    def init_weights(self):
        """
        If needed prunes and maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """
        # call pure
        if _init_weights:
            # Initialize weights
            self.apply(self._initialize_weights)

            # Tie weights should be skipped when not initializing all weights
            # since from_pretrained(...) calls tie weights anyways

            # TODO(wj-Mcat): enable all tie-weights later
            # self.tie_weights()

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            dtype (`paddle.dtype`, *optional*):
                Override the default `paddle.dtype` and load the model under this dtype.
        """
        dtype = kwargs.pop("dtype", None)

        if dtype is None:
            if config.dtype is not None:
                dtype = config.dtype
            else:
                dtype = paddle.get_default_dtype()

        with dtype_guard(dtype):
            model = cls(config, **kwargs)

        return model

    @classmethod
    def from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            dtype (`paddle.dtype`, *optional*):
                Override the default `paddle.dtype` and load the model under this dtype.
        """
        return cls._from_config(config, **kwargs)

    @classmethod
    def set_inference_config(cls, config, predictor_args, **kwargs):
        """
        All inference config can set here.
        Args:
            config : PretrainedConfig
                The config of the model.
            predictor_args : PredictorArgument
                The args of the predictor.
        """
        tensor_parallel_degree = kwargs.pop("tensor_parallel_degree", 1)
        tensor_parallel_rank = kwargs.pop("tensor_parallel_rank", 0)

        if predictor_args.mode == "dynamic" or predictor_args.speculate_method in [
            "eagle",
            "mtp",
        ]:
            config.tensor_parallel_degree = tensor_parallel_degree
            config.tensor_parallel_rank = tensor_parallel_rank
            config.model_name_or_path = predictor_args.model_name_or_path
            config.quant_type = predictor_args.quant_type
            config.cachekv_int8_type = predictor_args.cachekv_int8_type
            config.use_fake_parameter = predictor_args.use_fake_parameter
            config.single_card_ptq = not predictor_args.use_fake_parameter
        config.append_attn = predictor_args.append_attn
        config.decode_strategy = predictor_args.decode_strategy
        config.mla_use_matrix_absorption = predictor_args.mla_use_matrix_absorption
        config.weightonly_group_size = predictor_args.weightonly_group_size
        config.weight_block_size = predictor_args.weight_block_size
        config.moe_quant_type = predictor_args.moe_quant_type
        if config.quantization_config.quant_method is not None:
            predictor_args.weight_block_size = (
                config.quantization_config.weight_block_size
            )
            config.weight_block_size = predictor_args.weight_block_size

        if config.quantization_config.quant_type is not None:
            if predictor_args.mode == "dynamic":
                predictor_args.quant_type = config.quantization_config.quant_type
                config.quant_type = config.quantization_config.quant_type
            if "c8" in config.quant_type:
                predictor_args.cachekv_int8_type = "static"
                if predictor_args.mode == "dynamic":
                    config.cachekv_int8_type = "static"

            if predictor_args.mode == "dynamic":
                ptq_multicards_num = 0
                if os.path.exists(config.model_name_or_path):
                    prefix = "act_scales_"
                    for filename in os.listdir(config.model_name_or_path):
                        if filename.startswith(prefix):
                            ptq_multicards_num += 1

                logging.info(
                    f"PTQ from {ptq_multicards_num} cards, so we will not split"
                )
                if ptq_multicards_num > 1:
                    config.single_card_ptq = False

        if predictor_args.block_attn:
            config.block_size = predictor_args.block_size
            config.max_seq_len = predictor_args.total_max_length

        if predictor_args.speculate_method is not None:
            config.speculate_method = predictor_args.speculate_method
            config.speculate_max_draft_token_num = (
                predictor_args.speculate_max_draft_token_num
            )
            config.speculate_verify_window = predictor_args.speculate_verify_window
            config.speculate_max_candidate_len = (
                predictor_args.speculate_max_candidate_len
            )
            if predictor_args.speculate_method == "inference_with_reference":
                config.speculate_max_ngram_size = (
                    predictor_args.speculate_max_ngram_size
                )
            if predictor_args.speculate_method is not None:
                if not config.get("speculate_model_type", "None") in ["eagle", "mtp"]:
                    config.decode_strategy = "speculate_decoding"
        config.return_full_hidden_states = predictor_args.return_full_hidden_states

    @classmethod
    def confirm_inference_model(cls, predictor_args, **kwargs):
        """
        Confirm the inference model whether it need to change the AVX inference Model
        Args:
            model : PretrainedModel
                The model for inference.
            predictor_args : PredictorArgument
                The args of the predictor.
        """
        return cls

    @property
    def base_model(self):
        """
        PretrainedModel: The body of the same model architecture. It is the base
            model itself for base model or the base model attribute for derived
            model.
        """
        return getattr(self, self.base_model_prefix, self)

    @property
    def model_name_list(self):
        """
        list: Contains all supported built-in pretrained model names of the
            current PretrainedModel class.
        """
        # Todo: return all model name
        return list(self.pretrained_init_configuration.keys())

    def can_generate(self) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.
        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation
        if "GenerationMixin" in str(self.prepare_inputs_for_generation):
            return False
        return True

    def recompute_enable(self):
        r"""
        Enable Recompute.
        All layers with the `enable_recompute` attribute will be set to `True`
        """

        def fn(layer):
            if hasattr(layer, "enable_recompute") and (
                layer.enable_recompute is False or layer.enable_recompute == 0
            ):
                layer.enable_recompute = True

        self.apply(fn)

    def recompute_disable(self):
        r"""
        Disable Recompute.
        All layers with the `enable_recompute` attribute will be set to `False`
        """

        def fn(layer):
            if hasattr(layer, "enable_recompute") and (
                layer.enable_recompute is False or layer.enable_recompute == 0
            ):
                layer.enable_recompute = True

        self.apply(fn)

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        if self.config.tie_word_embeddings:
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()
            if output_embeddings is not None and input_embeddings is not None:
                if input_embeddings.weight.shape != output_embeddings.weight.shape:
                    logging.warning(
                        f"The shape of input embeddings is {input_embeddings.weight.shape} and the shape of output embeddings is {output_embeddings.weight.shape}. "
                        "This is only expected if you are calling the `resize_token_embeddings` method"
                    )
                output_embeddings.weight = input_embeddings.weight
                if getattr(output_embeddings, "bias", None) is not None:
                    # need to pad
                    if (
                        output_embeddings.weight.shape[0]
                        > output_embeddings.bias.shape[0]
                    ):
                        old_bias = output_embeddings.bias
                        pad_length = (
                            output_embeddings.weight.shape[0] - old_bias.shape[0]
                        )
                        output_embeddings.bias = output_embeddings.create_parameter(
                            shape=[output_embeddings.weight.shape[0]],
                            attr=output_embeddings._bias_attr,
                            dtype=output_embeddings._dtype,
                            is_bias=True,
                        )
                        new_bias = paddle.concat(
                            [
                                old_bias,
                                paddle.zeros(
                                    [pad_length], dtype=output_embeddings.bias.dtype
                                ),
                            ]
                        )
                        output_embeddings.bias.set_value(new_bias)
                    # need to trim
                    elif (
                        output_embeddings.weight.shape[0]
                        < output_embeddings.bias.shape[0]
                    ):
                        new_bias = output_embeddings.bias[
                            : output_embeddings.weight.shape[0]
                        ]
                        output_embeddings.bias = output_embeddings.create_parameter(
                            shape=[output_embeddings.weight.shape[0]],
                            attr=output_embeddings._bias_attr,
                            dtype=output_embeddings._dtype,
                            is_bias=True,
                        )
                        output_embeddings.bias.set_value(new_bias)

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """resize position embedding, this method should be overrited overwrited by downstream models

        Args:
            new_num_position_embeddings (int): the new position size

        Raises:
            NotImplementedError: when called and not be implemented
        """
        raise NotImplementedError(
            f"`resize_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `{self.__class__.__module__}.py`"
        )

    @classmethod
    def constructed_from_pretrained_config(cls, init_func=None) -> bool:
        """check if the model is constructed from `PretrainedConfig`
        Returns:
            bool: if the model is constructed from `PretrainedConfig`
        """
        return cls.config_class is not None and issubclass(
            cls.config_class, PretrainedConfig
        )

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model according to new_num_tokens.

        Args:
            new_num_tokens (Optional[int]):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or None, just
                returns a pointer to the input tokens embedding module of the model without doing anything.

        Returns:
            paddle.nn.Embedding: The input tokens Embeddings Module of the model.
        """
        old_embeddings: nn.Embedding = self.get_input_embeddings()
        if not new_num_tokens or new_num_tokens == old_embeddings.weight.shape[0]:
            return old_embeddings

        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        # 2. Update vocab_size
        self.base_model.config["vocab_size"] = new_num_tokens
        self.vocab_size = new_num_tokens

        # update init_config
        self._update_init_config(self.init_config, "vocab_size", new_num_tokens)

        # Tie the weights between the input embeddings and the output embeddings if needed.
        self.tie_weights()

        return new_embeddings

    def _update_init_config(self, init_config: dict, key: str, value: Any):
        """update init_config by <key, value> pair

        Args:
            init_config (dict): the init_config instance
            key (str): the key field
            value (Any): the new value of instance
        """
        if key in init_config:
            init_config[key] = value
            return

        for arg in init_config.get("init_args", []):
            if not isinstance(arg, PretrainedModel):
                continue
            self._update_init_config(arg.init_config, key, value)

    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (nn.Embedding):
                Old embeddings to be resized.
            new_num_tokens (Optional[int]):
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end.

        Returns:
            paddle.nn.Embedding: The resized Embedding Module or the old Embedding Module if new_num_tokens is None.
        """
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.shape
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that old_embeddings are an instance of"
                f" {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(
            new_num_tokens,
            old_embedding_dim,
            padding_idx=old_embeddings._padding_idx,
            sparse=old_embeddings._sparse,
        )

        # make sure that new_embeddings's dtype is same as the old embeddings' dtype
        if new_embeddings.weight.dtype != old_embeddings.weight.dtype:
            new_embeddings.to(dtype=old_embeddings.weight.dtype)

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        with paddle.no_grad():
            new_embeddings.weight[:n, :] = old_embeddings.weight[:n, :]

        return new_embeddings

    def __setattr__(self, name, value):
        value = adapt_stale_fwd_patch(self, name, value)
        return super(PretrainedModel, self).__setattr__(name, value)

    @classmethod
    def _resolve_model_file_path(
        cls: Type[PretrainedModel],
        pretrained_model_name_or_path: str,
        from_hf_hub: bool = False,
        from_aistudio: bool = False,
        cache_dir: str | None = None,
        subfolder: Optional[str] = "",
        config: PretrainedConfig = None,
        convert_from_torch: bool = False,
        use_safetensors: bool | None = None,
        variant=None,
    ) -> str:
        """resolve model target file path from `` and `cache_dir`

        1. when it is file path:
            return the weight file

        2. when it is model-name:
            2.1 check default `MODEL_HOME` + `model-mame` + model_state.pdparams
            2.2 get the url from `pretrained_resource_files_map`, and set it to `pretrained_model_name_or_path`

        3. when it is local dir:
            check whether the file<local_dir + weight_file> exist

        Args:
            cls (Type[PretrainedModel]): the inherited PretrainedModel class
            pretrained_model_name_or_path (str): the model-name/url/local_dir/local_dir
            cache_dir (Optional[str], optional): cache_dir is used when name_or_path is model-name/url. Defaults to None.
            convert_from_torch (bool, optional): whether support convert pytorch model to paddle model

        Returns:
            str: the model weight file path
        """
        is_sharded = False
        sharded_metadata = None

        if pretrained_model_name_or_path is not None:
            # the following code use a lot of os.path.join, hence setting subfolder to empty str if None
            if subfolder is None:
                subfolder = ""
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)

            def get_file_path(
                pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_NAME, variant
            ):
                return os.path.join(
                    pretrained_model_name_or_path,
                    subfolder,
                    _add_variant(SAFE_WEIGHTS_NAME, variant),
                )

            # pretrained_model_name_or_path is file
            if os.path.isfile(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
                is_local = True
            # pretrained_model_name_or_path is dir
            elif is_local:
                if use_safetensors is not False and os.path.isfile(
                    get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        SAFE_WEIGHTS_INDEX_NAME,
                        variant,
                    )
                ):
                    # Load from a sharded safetensors checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        SAFE_WEIGHTS_INDEX_NAME,
                        variant,
                    )
                    is_sharded = True
                elif use_safetensors is not False and os.path.isfile(
                    get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        SAFE_WEIGHTS_INDEX_NAME,
                        weight_name_suffix(),
                    )
                ):
                    # Load from a sharded safetensors checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        SAFE_WEIGHTS_INDEX_NAME,
                        weight_name_suffix(),
                    )
                    is_sharded = True
                elif use_safetensors is not False and os.path.isfile(
                    get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        SAFE_WEIGHTS_NAME,
                        variant,
                    )
                ):
                    # Load from a safetensors checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        SAFE_WEIGHTS_NAME,
                        variant,
                    )
                elif use_safetensors is not False and os.path.isfile(
                    get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        SAFE_WEIGHTS_NAME,
                        weight_name_suffix(),
                    )
                ):
                    # Load from a safetensors checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        SAFE_WEIGHTS_NAME,
                        weight_name_suffix(),
                    )
                elif os.path.isfile(
                    get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        PADDLE_WEIGHTS_INDEX_NAME,
                        variant,
                    )
                ):
                    # Load from a sharded PaddlePaddle checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        PADDLE_WEIGHTS_INDEX_NAME,
                        variant,
                    )
                    is_sharded = True
                elif os.path.isfile(
                    get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        PADDLE_WEIGHTS_INDEX_NAME,
                        weight_name_suffix(),
                    )
                ):
                    # Load from a sharded PaddlePaddle checkpoint for hybrid parallel model
                    archive_file = get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        PADDLE_WEIGHTS_INDEX_NAME,
                        weight_name_suffix(),
                    )
                    is_sharded = True
                elif os.path.isfile(
                    get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        PADDLE_WEIGHTS_NAME,
                        variant,
                    )
                ):
                    # Load from a PaddlePaddle checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        PADDLE_WEIGHTS_NAME,
                        variant,
                    )
                elif os.path.isfile(
                    get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        PADDLE_WEIGHTS_NAME,
                        weight_name_suffix(),
                    )
                ):
                    # Load from a PaddlePaddle checkpoint for hybrid parallel model
                    archive_file = get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        PADDLE_WEIGHTS_NAME,
                        weight_name_suffix(),
                    )
                elif os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(PYTORCH_WEIGHTS_INDEX_NAME, variant),
                    )
                ):
                    if from_hf_hub or convert_from_torch:
                        archive_file = os.path.join(
                            pretrained_model_name_or_path,
                            subfolder,
                            _add_variant(PYTORCH_WEIGHTS_INDEX_NAME, variant),
                        )
                    else:
                        raise ValueError(
                            f"Found {_add_variant(PYTORCH_WEIGHTS_INDEX_NAME, variant)} in directory"
                            f" {pretrained_model_name_or_path}. Please set convert_from_torch=True in from_pretrained. eg, Model.from_pretrained(model_name, convert_from_torch=True) "
                        )
                elif os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(PYTORCH_WEIGHTS_NAME, variant),
                    )
                ):
                    if from_hf_hub or convert_from_torch:
                        archive_file = os.path.join(
                            pretrained_model_name_or_path,
                            subfolder,
                            _add_variant(PYTORCH_WEIGHTS_NAME, variant),
                        )
                    else:
                        raise ValueError(
                            f"Found {_add_variant(PYTORCH_WEIGHTS_NAME, variant)} in directory"
                            f" {pretrained_model_name_or_path}. Please set convert_from_torch=True in from_pretrained. eg, Model.from_pretrained(model_name, convert_from_torch=True) "
                        )
                else:
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(PADDLE_WEIGHTS_NAME, variant)}, found in directory"
                        f" {pretrained_model_name_or_path}."
                    )

            elif pretrained_model_name_or_path in cls.pretrained_init_configuration:
                # fetch the weight url from the `pretrained_resource_files_map`
                resource_file_url = cls.pretrained_resource_files_map["model_state"][
                    pretrained_model_name_or_path
                ]
                resolved_archive_file = resolve_file_path(
                    pretrained_model_name_or_path,
                    [resource_file_url],
                    subfolder,
                    cache_dir=cache_dir,
                    from_aistudio=from_aistudio,
                    from_hf_hub=from_hf_hub,
                )
            else:
                if use_safetensors is True:
                    filenames = [
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                        _add_variant(SAFE_WEIGHTS_NAME, variant),
                    ]
                elif use_safetensors is None:
                    filenames = [
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                        _add_variant(PADDLE_WEIGHTS_INDEX_NAME, variant),
                        _add_variant(SAFE_WEIGHTS_NAME, variant),
                        _add_variant(PADDLE_WEIGHTS_NAME, variant),
                        _add_variant(PYTORCH_WEIGHTS_INDEX_NAME, variant),
                        _add_variant(PYTORCH_WEIGHTS_NAME, variant),
                    ]
                else:
                    filenames = [
                        _add_variant(PADDLE_WEIGHTS_INDEX_NAME, variant),
                        _add_variant(PADDLE_WEIGHTS_NAME, variant),
                        _add_variant(PYTORCH_WEIGHTS_INDEX_NAME, variant),
                        _add_variant(PYTORCH_WEIGHTS_NAME, variant),
                    ]
                resolved_archive_file = resolve_file_path(
                    pretrained_model_name_or_path,
                    filenames,
                    subfolder,
                    cache_dir=cache_dir,
                    from_aistudio=from_aistudio,
                    from_hf_hub=from_hf_hub,
                )
                if resolved_archive_file is None:
                    raise EnvironmentError(
                        f"Error no files {filenames} found in repo {pretrained_model_name_or_path}."
                    )
                elif "pytorch_model.bin" in str(resolved_archive_file):
                    if not from_hf_hub and not convert_from_torch:
                        raise ValueError(
                            f"Download pytorch weight in "
                            f" {resolved_archive_file}. Please set convert_from_torch=True in from_pretrained. eg, Model.from_pretrained(model_name, convert_from_torch=True) "
                        )

            if is_local:
                logging.info(f"Loading weights file {archive_file}")
                resolved_archive_file = archive_file
            else:
                logging.info(
                    f"Loading weights file from cache at {resolved_archive_file}"
                )
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        resolved_sharded_files = None
        if str(resolved_archive_file).endswith(".json"):
            is_sharded = True
        if is_sharded:
            # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_sharded_files, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                from_aistudio=from_aistudio,
                from_hf_hub=from_hf_hub,
                cache_dir=cache_dir,
                subfolder=subfolder,
            )

        return (
            resolved_archive_file,
            resolved_sharded_files,
            sharded_metadata,
            is_sharded,
        )

    @classmethod
    def _load_pretrained_model(
        cls,
        model: PretrainedModel,
        state_dict: Dict[str, Tensor],
        loaded_keys: List[str],
        resolved_archive_file: Union[str, List] = [],
        pretrained_model_name_or_path=None,
        config=None,
        ignore_mismatched_sizes=False,
        low_cpu_mem_usage=False,
        dtype=None,
        keep_in_fp32_modules=None,
        quantization_linear_list=None,
        sharded_metadata=None,
    ) -> Tuple[List[str]]:
        """load the state_dict into model, and do the following things:

            * check the

        Args:
            model (PretrainedModel): the pretrained model instance
            state_dict (Dict[str, Tensor]): the model state dict data
            loaded_keys (List[str]):
            ignore_mismatched_sizes (bool, optional): whether ignore error when tensor size mismatched. Defaults to False.
            dtype (_type_, optional): the dtype of model state dict. Defaults to None.

        Returns:
            Tuple[List[str]]: _description_
        """
        is_safetensors = False

        model_state_dict = model.state_dict()

        expected_keys = list(model_state_dict.keys())
        prefix = model.base_model_prefix

        if len(prefix) > 0:
            has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
            expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
        else:
            has_prefix_module = False
            expects_prefix_module = False

        # key re-naming operations are never done on the keys
        # that are loaded, but always on the keys of the newly initialized model
        remove_prefix_from_model = not has_prefix_module and expects_prefix_module
        add_prefix_to_model = has_prefix_module and not expects_prefix_module

        if remove_prefix_from_model:
            _prefix = f"{prefix}."
            expected_keys_not_prefixed = [
                s for s in expected_keys if not s.startswith(_prefix)
            ]
            expected_keys = [
                s[len(_prefix) :] if s.startswith(_prefix) else s for s in expected_keys
            ]
            if quantization_linear_list is not None:
                quantization_linear_list = [
                    s[len(_prefix) :] if s.startswith(_prefix) else s
                    for s in quantization_linear_list
                ]
        elif add_prefix_to_model:
            expected_keys = [".".join([prefix, s]) for s in expected_keys]
            if quantization_linear_list is not None:
                quantization_linear_list = [
                    ".".join([prefix, s]) for s in quantization_linear_list
                ]

        # Weight quantization if not yet quantized & update loaded_keys
        if (
            hasattr(config, "quantization_config")
            and config.quantization_config.is_weight_quantize()
        ):
            try:
                from ..quantization.quantization_utils import (
                    convert_to_quantize_state_dict,
                    update_loaded_state_dict_keys,
                )
            except ImportError:
                raise ImportError(
                    "Quantization features require `paddlepaddle >= 2.5.2`"
                )
            if state_dict is not None:
                state_dict = convert_to_quantize_state_dict(
                    state_dict,
                    quantization_linear_list,
                    config.quantization_config,
                    dtype,
                )
                loaded_keys = [k for k in state_dict.keys()]
            else:
                loaded_keys = update_loaded_state_dict_keys(
                    loaded_keys, quantization_linear_list, config.quantization_config
                )
            if keep_in_fp32_modules is None:
                keep_in_fp32_modules = (
                    ["quant_scale"]
                    if config.quantization_config.weight_quantize_algo in ["nf4", "fp4"]
                    else None
                )
            else:
                keep_in_fp32_modules = (
                    keep_in_fp32_modules + ["quant_scale"]
                    if config.quantization_config.weight_quantize_algo in ["nf4", "fp4"]
                    else keep_in_fp32_modules
                )

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        # Optimize for skip unused shard files for supper large model
        if sharded_metadata is not None:
            assert isinstance(resolved_archive_file, list)
            new_archive_file = []
            skip_archive_file = []
            expected_keys_set = set(expected_keys)
            for file in resolved_archive_file:
                filename = os.path.split(file)[-1]
                if not expected_keys_set.isdisjoint(
                    set(sharded_metadata["file_map"][filename])
                ):
                    new_archive_file.append(file)
                else:
                    skip_archive_file.append(filename)

            resolved_archive_file = new_archive_file
            if len(skip_archive_file) > 0:
                logging.info(
                    f"Skip load files for not contrains expected key, {skip_archive_file}"
                )

        # Some models may have keys that are not in the state by design, removing them before needlessly warning
        # the user.
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [
                    k for k in unexpected_keys if re.search(pat, k) is None
                ]

        # Set some modules to fp32 if any
        if keep_in_fp32_modules is not None:
            for name, param in model.named_parameters():
                if any(
                    module_to_keep_in_fp32 in name
                    for module_to_keep_in_fp32 in keep_in_fp32_modules
                ):
                    if param.dtype != paddle.float32:
                        param_fp32 = param.cast(dtype=paddle.float32)
                        param_fp32_tensor = param_fp32.value().get_tensor()
                        param_tensor = param.value().get_tensor()
                        param_tensor._share_data_with(param_fp32_tensor)

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if (
            len(cls.base_model_prefix) > 0
            and not hasattr(model, cls.base_model_prefix)
            and has_prefix_module
        ):
            start_prefix = cls.base_model_prefix + "."
        if (
            len(cls.base_model_prefix) > 0
            and hasattr(model, cls.base_model_prefix)
            and not has_prefix_module
        ):
            model_to_load = getattr(model, cls.base_model_prefix)
            base_model_expected_keys = list(model_to_load.state_dict().keys())
            if any(
                key in expected_keys_not_prefixed
                and key not in base_model_expected_keys
                for key in loaded_keys
            ):
                raise ValueError(
                    "The state dictionary of the model you are trying to load is corrupted. Are you sure it was "
                    "properly saved?"
                )

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    # If the checkpoint is sharded, we may not have the key here.
                    if checkpoint_key not in state_dict:
                        continue
                    model_key = checkpoint_key
                    if remove_prefix_from_model:
                        # The model key starts with `prefix` but `checkpoint_key` doesn't so we add it.
                        model_key = f"{prefix}.{checkpoint_key}"
                    elif add_prefix_to_model:
                        # The model key doesn't start with `prefix` but `checkpoint_key` does so we remove it.
                        model_key = ".".join(checkpoint_key.split(".")[1:])

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape
                        != model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (
                                checkpoint_key,
                                state_dict[checkpoint_key].shape,
                                model_state_dict[model_key].shape,
                            )
                        )
                        del state_dict[checkpoint_key]
            return mismatched_keys

        def _fuse_or_split_keys(
            state_dict,
            config,
            loaded_keys,
            pre_tensor_parallel_split=False,
            resume_state_dict=None,
        ):
            if resume_state_dict is not None:
                state_dict.update(resume_state_dict)

            before_fuse_keys = list(state_dict.keys())
            if pre_tensor_parallel_split:
                tp_actions = cls.get_tensor_parallel_convert_actions(
                    config, loaded_keys, ignore_error=True
                )
            else:
                tp_actions = None
            state_dict, resume_state_dict = cls.convert_fuse_and_split(
                config, state_dict, tp_actions
            )
            after_fuse_keys = list(state_dict.keys())

            fused_keys = list(set(before_fuse_keys) - set(after_fuse_keys))
            new_keys = list(set(after_fuse_keys) - set(before_fuse_keys))

            return state_dict, resume_state_dict, fused_keys, new_keys

        if state_dict is not None:
            # have loaded all state_dict, no resume state_dict
            state_dict, _, fused_keys, new_keys = _fuse_or_split_keys(
                state_dict,
                config,
                loaded_keys,
                pre_tensor_parallel_split=(
                    True
                    if config is not None and config.tensor_parallel_degree > 1
                    else False
                ),
            )
            missing_keys = list(set(missing_keys) - set(new_keys))
            unexpected_keys = list(set(unexpected_keys) - set(fused_keys))

            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                loaded_keys,
                add_prefix_to_model,
                remove_prefix_from_model,
                ignore_mismatched_sizes,
            )

            if (
                hasattr(config, "quantization_config")
                and config.quantization_config.is_weight_quantize()
            ):
                error_msgs = _load_state_dict_into_meta_model(
                    model_to_load,
                    state_dict,
                    loaded_keys,
                    start_prefix,
                    expected_keys,
                    dtype=dtype,
                    is_safetensors=is_safetensors,
                    keep_in_fp32_modules=keep_in_fp32_modules,
                )
            else:
                error_msgs = _load_state_dict_into_model(
                    model_to_load, state_dict, start_prefix
                )
        else:
            # Sharded checkpoint or whole but low_cpu_mem_usage==True

            # This should always be a list but, just to be sure.
            if not isinstance(resolved_archive_file, list):
                resolved_archive_file = [resolved_archive_file]

            error_msgs = []
            mismatched_keys = []
            resume_state_dict = {}

            for shard_file in resolved_archive_file:
                pre_tensor_parallel_split = False
                if (
                    shard_file.endswith(".safetensors")
                    and config.tensor_parallel_degree > 1
                    and "tp" not in os.path.split(shard_file)[-1]
                ):
                    pre_tensor_parallel_split = True
                    assert loaded_keys is not None, "loaded_keys is not None."
                    tp_actions = cls.get_tensor_parallel_convert_actions(
                        config, loaded_keys, ignore_error=True
                    )
                # Here we use expected_keys to optimize weights loading for pipeline model. Only works for safetensors
                filter_dict_keys = set(expected_keys)
                fuse_actions, _ = cls.get_fuse_or_split_param_convert_actions(
                    config, loaded_keys, is_fuse=True
                )
                split_actions, _ = cls.get_fuse_or_split_param_convert_actions(
                    config, loaded_keys, is_fuse=False
                )
                for k in list(fuse_actions.keys()):
                    need_add_except_key = k[-1] in expected_keys
                    if need_add_except_key:
                        filter_dict_keys |= set(k[:-1])
                    # remove pre_tensor_parallel_split function from tp_actions
                    if pre_tensor_parallel_split:
                        for item in k[:-1]:
                            if item in tp_actions:
                                tp_actions.pop(item, None)

                for k in list(split_actions.keys()):
                    need_add_except_key = False
                    for item in k[:-1]:
                        if item in expected_keys:
                            need_add_except_key = True
                            break
                    if need_add_except_key:
                        filter_dict_keys.add(k[-1])
                    # remove pre_tensor_parallel_split function from tp_actions
                    if pre_tensor_parallel_split:
                        if k[-1] in tp_actions:
                            fuse_actions.pop(k[-1], None)

                if config.quantization_config.is_weight_quantize():
                    filter_dict_keys = None
                state_dict = load_state_dict(
                    shard_file,
                    tp_actions if pre_tensor_parallel_split else None,
                    filter_dict_keys,
                )

                # convert for fusing or splitting weights
                state_dict, resume_state_dict, fused_keys, new_keys = (
                    _fuse_or_split_keys(
                        state_dict,
                        config,
                        loaded_keys,
                        pre_tensor_parallel_split=pre_tensor_parallel_split,
                        resume_state_dict=resume_state_dict,
                    )
                )
                missing_keys = list(set(missing_keys) - set(new_keys))
                unexpected_keys = list(set(unexpected_keys) - set(fused_keys))

                if config.quantization_config.is_weight_quantize():
                    state_dict = convert_to_quantize_state_dict(
                        state_dict,
                        quantization_linear_list,
                        config.quantization_config,
                        dtype,
                    )

                # Mismatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
                # matching the weights in the model.
                mismatched_keys += _find_mismatched_keys(
                    state_dict,
                    model_state_dict,
                    loaded_keys,
                    add_prefix_to_model,
                    remove_prefix_from_model,
                    ignore_mismatched_sizes,
                )

                if (
                    config.tensor_parallel_degree > 1
                    and ".tp" not in shard_file
                    and not pre_tensor_parallel_split
                ):
                    logging.info("Converting state_dict to Tensor Parallel Format")
                    # ignore error for multi shard, since only parts of data
                    state_dict = cls.convert_tensor_parallel(
                        None,
                        config,
                        state_dict=state_dict,
                        ignore_error=len(resolved_archive_file) > 1,
                    )
                    logging.info("Converted state_dict to Tensor Parallel Format")

                if low_cpu_mem_usage or config.quantization_config.is_weight_quantize():
                    new_error_msgs = _load_state_dict_into_meta_model(
                        model_to_load,
                        state_dict,
                        loaded_keys,
                        start_prefix,
                        expected_keys,
                        dtype=dtype,
                        is_safetensors=is_safetensors,
                        keep_in_fp32_modules=keep_in_fp32_modules,
                    )
                    error_msgs += new_error_msgs
                else:
                    error_msgs += _load_state_dict_into_model(
                        model_to_load, state_dict, start_prefix
                    )

                # force memory release
                del state_dict
                gc.collect()

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if " but the expected shape is" in error_msg:
                error_msg += "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
            raise RuntimeError(
                f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}"
            )

        if len(unexpected_keys) > 0:
            if logging.logging.level < 20:
                logging.warning(
                    f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                    f" initializing {model.__class__.__name__}: {sorted(unexpected_keys)}\n- This IS expected if you are"
                    f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                    " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                    " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                    f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                    " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
                )
            else:
                logging.warning(
                    f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                    f" initializing the model, - This IS expected if you are"
                    f" initializing the model from a checkpoint of a model trained on another task or"
                    " with another architecture."
                )
        else:
            logging.info(
                f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n"
            )

        if len(missing_keys) > 0:
            logging.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logging.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logging.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Creates an instance of `PretrainedModel`. Model weights are loaded
        by specifying name of a built-in pretrained model, a pretrained model from HF Hub, a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model or dir path
                to load from. The string can be:

                - Name of a built-in pretrained model
                - Name of a pretrained model from HF Hub
                - Name of a community-contributed pretrained model.
                - Local directory path which contains model weights file("model_state.pdparams")
                    and model config file ("model_config.json").
            from_hf_hub (bool): load model from huggingface hub. Default to `False`.
            subfolder (str, optional) An optional value corresponding to a folder inside the repo.
                Only works when loading from Huggingface Hub.
            *args (tuple): Position arguments for model `__init__`. If provided,
                use these as position argument values for model initialization.
            **kwargs (dict): Keyword arguments for model `__init__`. If provided,
                use these to update pre-defined keyword argument values for model
                initialization. If the keyword is in `__init__` argument names of
                base model, update argument values of the base model; else update
                argument values of derived model.
            load_state_as_np (bool, optional): The weights read in can be choosed
                to place on CPU or GPU though the model is on the default device.
                If `True`, load the model weights as `numpy.ndarray` on CPU.
                Otherwise, weights would be loaded as tensors on the default
                device. Note that if on GPU, the latter would creates extra
                temporary tensors in addition to the model weights, which
                doubles the memory usage . Thus it is suggested to use `True`
                for big models on GPU. Default to `False`.

        Returns:
            PretrainedModel: An instance of `PretrainedModel`.

        Example:
            .. code-block::

                from paddlenlp.transformers import BertForSequenceClassification

                # Name of built-in pretrained model
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

                # Name of pretrained model from PaddleHub
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

                # Name of community-contributed pretrained model
                model = BertForSequenceClassification.from_pretrained('yingyibiao/bert-base-uncased-sst-2-finetuned', num_labels=3)

                # Load from local directory path
                model = BertForSequenceClassification.from_pretrained('./my_bert/')
        """
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.get("force_download", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        dtype = kwargs.pop("dtype", None)
        from_hf_hub = kwargs.pop("from_hf_hub", False)
        from_aistudio = kwargs.pop("from_aistudio", False)
        subfolder = kwargs.pop("subfolder", None)
        if subfolder is None:
            subfolder = ""
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop(
            "use_safetensors", None if is_safetensors_available() else False
        )

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
        convert_from_torch = kwargs.pop("convert_from_torch", None)
        load_state_as_np = kwargs.pop("load_state_as_np", None)
        if load_state_as_np is not None:
            logging.warning("`load_state_as_np` is deprecated,  please delete it!")

        model_kwargs = kwargs

        if convert_from_torch is None and os.environ.get("from_modelscope", False):
            logging.warning(
                "If you are attempting to load weights from ModelScope Hub and want to disable the default behavior of considering torch weights,"
                " you can set ·convert_from_torch=False·. By default, `convert_from_torch` is set to `True`. "
            )
            convert_from_torch = True

        # from_hf_hub default enable convert_from_torch
        if from_hf_hub and convert_from_torch is None:
            logging.warning(
                "If you are attempting to load weights from Hugging Face Hub and want to disable the default behavior of considering torch weights,"
                " you can set ·convert_from_torch=False·. By default, `convert_from_torch` is set to `True`. "
            )
            convert_from_torch = True
        # convert_from_torch default is False
        if convert_from_torch is None:
            convert_from_torch = False

        # 1. get the PretrainedConfig to init model
        if not isinstance(config, PretrainedConfig):
            config_path = (
                config if config is not None else pretrained_model_name_or_path
            )
            config, model_kwargs = (
                cls.config_class.from_pretrained(  # NOTE cls.config_class : Qwen2VLForConditionalGeneration
                    config_path,
                    cache_dir=cache_dir,
                    from_hf_hub=from_hf_hub,
                    from_aistudio=from_aistudio,
                    subfolder=subfolder,
                    return_unused_kwargs=True,
                    **kwargs,
                )
            )
        if "from_aistudio" in model_kwargs:
            model_kwargs.pop("from_aistudio")

        if dtype is None:
            dtype = config.dtype
        config.dtype = dtype

        init_contexts = []

        if dtype:
            init_contexts.append(dtype_guard(dtype))

        # Keep in fp32 modules
        keep_in_fp32_modules = None
        use_keep_in_fp32_modules = False

        # resolve model_weight file
        resolved_archive_file, resolved_sharded_files, sharded_metadata, is_sharded = (
            cls._resolve_model_file_path(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                subfolder=subfolder,
                from_hf_hub=from_hf_hub,
                from_aistudio=from_aistudio,
                config=config,
                convert_from_torch=False,
                use_safetensors=use_safetensors,
                variant=variant,
            )
        )

        if convert_from_torch and state_dict is None:
            if (
                resolved_archive_file.endswith(PYTORCH_WEIGHTS_NAME)
                or resolved_archive_file.endswith(PYTORCH_WEIGHTS_INDEX_NAME)
                or resolved_archive_file.endswith(SAFE_WEIGHTS_NAME)
                or resolved_archive_file.endswith(SAFE_WEIGHTS_INDEX_NAME)
            ):
                # try to get the name-mapping info
                convert_dir = os.path.dirname(resolved_archive_file)
                logging.info(
                    f"Starting to convert pytorch weight file<{resolved_archive_file}> to "
                    f"paddle weight file<{convert_dir}> ..."
                )
                state_dict = cls.convert(
                    resolved_archive_file,
                    config,
                    # cache_dir=os.path.join(cache_dir, pretrained_model_name_or_path, subfolder),
                    cache_dir=convert_dir,
                )
            elif (
                resolved_archive_file.endswith(PADDLE_WEIGHTS_NAME)
                or resolved_archive_file.endswith(PADDLE_WEIGHTS_INDEX_NAME)
                or resolved_archive_file.endswith(".pdparams")
            ):
                print(f"file: {resolved_archive_file} is paddle weight.")
            else:
                raise ValueError(
                    f"Unexpected file: {resolved_archive_file} for weight conversion."
                )
            # load pt weights early so that we know which dtype to init the model under
        if not is_sharded and state_dict is None:
            # 4. loading non-sharded ckpt from the state dict
            if config.tensor_parallel_degree > 1 and resolved_archive_file.endswith(
                "model_state.pdparams"
            ):
                state_dict = cls.convert_tensor_parallel(resolved_archive_file, config)
            elif config.tensor_parallel_degree > 1 and resolved_archive_file.endswith(
                "model.safetensors"
            ):
                raise NotImplementedError
            else:
                state_dict = load_state_dict(resolved_archive_file)

            logging.info("Loaded weights file from disk, setting weights to model.")

        # Check if `_keep_in_fp32_modules` is not None
        use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and (
            dtype == "float16" or dtype == "bfloat16"
        )

        if state_dict is not None:
            loaded_state_dict_keys = [k for k in state_dict.keys()]
            # will only support load paddle.Tensor to model.
            for k in list(state_dict.keys()):
                if not isinstance(state_dict[k], paddle.Tensor):
                    with device_guard():
                        state_dict[k] = paddle.Tensor.__call__(
                            state_dict.pop(k), zero_copy=True
                        )
        else:
            if is_sharded:
                loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
            else:
                loaded_state_dict_keys = [k for k in state_dict.keys()]

        if low_cpu_mem_usage:  # or use_keep_in_fp32_modules:
            state_dict = None

        # will only support load paddle.Tensor to model.
        if state_dict is not None:
            for k in list(state_dict.keys()):
                if not isinstance(state_dict[k], paddle.Tensor):
                    with device_guard():
                        state_dict[k] = paddle.Tensor.__call__(
                            state_dict.pop(k), zero_copy=True
                        )
        # 3. init the model
        init_args = config["init_args"] or ()
        with ContextManagers(init_contexts):
            model = cls(config, *init_args, **model_kwargs)

        if use_keep_in_fp32_modules:
            # low_cpu_mem_usage = True
            keep_in_fp32_modules = model._keep_in_fp32_modules
        else:
            keep_in_fp32_modules = []

        quantization_linear_list = None

        model, missing_keys, unexpected_keys, mismatched_keys = (
            cls._load_pretrained_model(
                model=model,
                state_dict=state_dict,
                loaded_keys=loaded_state_dict_keys,
                resolved_archive_file=(
                    resolved_sharded_files if is_sharded else resolved_archive_file
                ),
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                config=config,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                low_cpu_mem_usage=low_cpu_mem_usage,
                dtype=dtype,
                keep_in_fp32_modules=keep_in_fp32_modules,
                quantization_linear_list=quantization_linear_list,
                sharded_metadata=sharded_metadata if is_sharded else None,
            )
        )

        # load generation_config.json
        if model.can_generate() and pretrained_model_name_or_path is not None:
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    from_hf_hub=from_hf_hub,
                    from_aistudio=from_aistudio,
                    subfolder=subfolder,
                    **kwargs,
                )
            except:
                logging.info(
                    "Generation config file not found, using a generation config created from the model config."
                )
                pass

        # Note:
        # 1. PipelineLayer will create parameters for each layer and
        # call `_synchronize_shared_weights()` to synchronize the shared parameters.
        # 2. When setting the model `state_dict`, `_synchronize_shared_weights` will be called to
        # synchronize the shared parameters.
        # However, when state dict only contains the one piece of shared parameters, the shared parameters
        # will be different from the original shared parameters.

        if isinstance(model, PipelineLayer):
            model._synchronize_shared_weights()

        if paddle.in_dynamic_mode():
            return model

        return model, state_dict

    def merge_auto_dist_configs(self, configs):
        """
        Merged all auto dist configs into one config.
        configs is a list of config,every config is a dict,which means a model auto_dist_config.
        [
            {
                mp_config (dict): {
                    "parallelize_plan": dict, the plan to shard the layer.
                }
                pp_config (dict): {
                    "split_spec": OrderedDict|dict|str|list(str), The pipeline parallel split point.
                    "global_spec": str|list(str), make the output tensor of specific layers on global mesh.
                }
            },{
                mp_config (dict): {
                    "parallelize_plan": dict, the plan to shard the layer.
                }
                pp_config (dict): {
                    "split_spec": OrderedDict|dict|str|list(str), The pipeline parallel split point.
                    "global_spec": str|list(str), make the output tensor of specific layers on global mesh.
                }
            },....
        ]
        """
        assert isinstance(configs, (dict, list))
        if isinstance(configs, dict):
            return configs
        final_config = {
            "mp_config": None,
            "sp_config": None,
            "pp_config": None,
        }
        for config in configs:
            if "mp_config" in config and config["mp_config"] is not None:
                if final_config["mp_config"] is None:
                    final_config["mp_config"] = config["mp_config"]
                else:
                    for k, v in config["mp_config"]["parallelize_plan"].items():
                        assert (
                            k
                            not in final_config["mp_config"]["parallelize_plan"].keys()
                        ), f"sublayer mp_config should be a subset of model but got sublayer config {config['mp_config']} and model config {final_config['mp_config']}."
                        final_config["mp_config"]["parallelize_plan"][k] = v
            if "sp_config" in config and config["sp_config"] is not None:
                if final_config["sp_config"] is None:
                    final_config["sp_config"] = config["sp_config"]
                else:
                    for k, v in config["sp_config"]["parallelize_plan"].items():
                        assert (
                            k
                            not in final_config["sp_config"]["parallelize_plan"].keys()
                        ), f"sublayer sp_config should be a subset of model but got sublayer config {config['sp_config']} and model config {final_config['sp_config']}."
                        final_config["sp_config"]["parallelize_plan"][k] = v
            if "pp_config" in config and config["pp_config"] is not None:
                if isinstance(config["pp_config"]["split_spec"], str):
                    config["pp_config"]["split_spec"] = [
                        config["pp_config"]["split_spec"]
                    ]
                    if final_config["pp_config"] is None:
                        final_config["pp_config"] = config["pp_config"]
                    else:
                        final_config["pp_config"]["split_spec"] += config["pp_config"][
                            "split_spec"
                        ]
                elif isinstance(config["pp_config"]["split_spec"], (tuple, list)):
                    if final_config["pp_config"] is None:
                        final_config["pp_config"] = config["pp_config"]
                    else:
                        final_config["pp_config"]["split_spec"] += config["pp_config"][
                            "split_spec"
                        ]

        if (
            final_config["pp_config"] is not None
            and len(final_config["pp_config"]["split_spec"]) == 1
        ):
            final_config["pp_config"]["split_spec"] = final_config["pp_config"][
                "split_spec"
            ][0]

        return final_config

    def _generate_auto_dist_config(self, auto_dist_degree):
        merged_config = {
            "sp_config": None,
            "mp_config": None,
            "pp_config": None,
        }
        for name, layer in self.named_sublayers(include_self=True):
            if hasattr(layer, "auto_dist_config"):
                if name != "":
                    prefix = name + "."
                else:
                    prefix = ""
                layer_config = layer.auto_dist_config(prefix)
                merged_config = self.merge_auto_dist_configs(
                    [merged_config, layer_config]
                )
                for _, deeper_layer in layer.named_sublayers():
                    if hasattr(deeper_layer, "auto_dist_config"):
                        # mask all `auto_dist_config` methods in deeper layer
                        deeper_layer.auto_dist_config = lambda x: {}

        final_config = {
            "dp_config": None,
            "mp_config": None,
            "pp_config": None,
        }

        if (
            "tensor_parallel" in auto_dist_degree
            and auto_dist_degree["tensor_parallel"]
        ):
            merged_config["mp_config"] is not None
            final_config["mp_config"] = merged_config["mp_config"]

        if (
            "sequence_parallel" in auto_dist_degree
            and auto_dist_degree["sequence_parallel"]
        ):
            merged_config["sp_config"] is not None
            final_config["mp_config"] = merged_config["sp_config"]

        if (
            "pipeline_parallel" in auto_dist_degree
            and auto_dist_degree["pipeline_parallel"]
        ):
            merged_config["pp_config"] is not None
            final_config["pp_config"] = merged_config["pp_config"]

        return final_config
