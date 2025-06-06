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

import json
import os
import shutil
from functools import lru_cache
from typing import Dict, Optional, Union

import numpy as np

from .....utils.deps import class_requires_deps
from .tokenizer_utils import PretrainedTokenizer
from .tokenizer_utils_base import (
    AddedToken,
    BatchEncoding,
    EncodedInput,
    PaddingStrategy,
)

__all__ = [
    "GPTTokenizer",
]


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    _chr = chr
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [_chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


@class_requires_deps("regex")
class GPTTokenizer(PretrainedTokenizer):
    """
    Constructs a GPT tokenizer based on byte-level Byte-Pair-Encoding.

    This tokenizer inherits from :class:`~paddlenlp.transformers.tokenizer_utils.PretrainedTokenizer`
    which contains most of the main methods. For more information regarding those methods,
    please refer to this superclass.

    Args:
        vocab_file (str):
            Path to the vocab file.
            The vocab file contains a mapping from vocabulary strings to indices.
        merges_file (str):
            Path to the merge file.
            The merge file is used to split the input sentence into "subword" units.
            The vocab file is then used to encode those units as intices.
        errors (str):
            Paradigm to follow when decoding bytes to UTF-8.
            Defaults to `'replace'`.
        max_len (int, optional):
            The maximum value of the input sequence length.
            Defaults to `None`.

    Examples:
        .. code-block::

            from paddlenlp.transformers import GPTTokenizer

            tokenizer = GPTTokenizer.from_pretrained('gpt2-medium-en')
            print(tokenizer('Welcome to use PaddlePaddle and PaddleNLP'))

            '''
            {'input_ids': [14618, 284, 779, 350, 37382, 47, 37382, 290, 350, 37382, 45, 19930],
            'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
            '''

    """

    resource_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
    }  # for save_pretrained
    gpt_vocab_link = (
        "http://bj.bcebos.com/paddlenlp/models/transformers/gpt/gpt-en-vocab.json"
    )
    gpt_merges_link = (
        "http://bj.bcebos.com/paddlenlp/models/transformers/gpt/gpt-en-merges.txt"
    )
    pretrained_resource_files_map = {
        "vocab_file": {
            "gpt3-175B-en": gpt_vocab_link,
            "gpt3-89B-en": gpt_vocab_link,
            "gpt3-13B-en": gpt_vocab_link,
            "gpt3-6.7B-en": gpt_vocab_link,
            "gpt3-1.3B-en": gpt_vocab_link,
            "gpt2-xl-en": gpt_vocab_link,
            "gpt2-large-en": gpt_vocab_link,
            "gpt2-medium-en": gpt_vocab_link,
            "gpt2-en": gpt_vocab_link,
            "gpt2-small-en": gpt_vocab_link,
        },
        "merges_file": {
            "gpt3-175B-en": gpt_merges_link,
            "gpt3-89B-en": gpt_merges_link,
            "gpt3-13B-en": gpt_merges_link,
            "gpt3-6.7B-en": gpt_merges_link,
            "gpt3-1.3B-en": gpt_merges_link,
            "gpt2-xl-en": gpt_merges_link,
            "gpt2-large-en": gpt_merges_link,
            "gpt2-medium-en": gpt_merges_link,
            "gpt2-en": gpt_merges_link,
            "gpt2-small-en": gpt_merges_link,
        },
    }
    pretrained_init_configuration = {
        "gpt3-175B-en": {},
        "gpt3-89B-en": {},
        "gpt3-13B-en": {},
        "gpt3-6.7B-en": {},
        "gpt3-1.3B-en": {},
        "gpt2-xl-en": {},
        "gpt2-large-en": {},
        "gpt2-medium-en": {},
        "gpt2-en": {},
        "gpt2-small-en": {},
    }

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        max_len=None,
        pad_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        eol_token="\u010a",
        add_prefix_space=False,
        add_bos_token=False,
        **kwargs  # The token of newline.
    ):
        import regex as re

        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False)
            if isinstance(pad_token, str)
            else pad_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False)
            if isinstance(unk_token, str)
            else unk_token
        )
        self.eol_token = eol_token
        self._build_special_tokens_map_extended(
            bos_token=(
                pad_token
                if getattr(self, "bos_token", None) is None
                else self.bos_token
            ),
            eos_token=eos_token,
            unk_token=unk_token,
        )

        self._vocab_file = vocab_file
        self._merges_file = merges_file
        self.max_len = max_len if max_len is not None else int(1e12)
        self.num_command_tokens = 2
        self.num_type_tokens = 2

        with open(vocab_file, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)

        self.decoder = {v: k for k, v in self.encoder.items()}

        self.num_tokens = len(self.encoder)
        self.num_text_tokens = self.num_tokens - 1
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        with open(merges_file, encoding="utf-8") as f:
            bpe_data = f.read().split("\n")[1:-1]

        bpe_merges = [tuple(merge.split()) for merge in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.add_prefix_space = add_prefix_space
        self.add_bos_token = add_bos_token

        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    @property
    def vocab_size(self):
        """
        Returns the size of vocabulary.

        Returns:
            int: The sum of size of vocabulary and the size of special tokens.

        """

        return len(self.encoder)

    @property
    def eol_token_id(self):
        if self.eol_token is None:
            return None
        return self.convert_tokens_to_ids(self.eol_token)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """Tokenize a string."""
        import regex as re

        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.decoder[index]

    def convert_ids_to_string(self, ids):
        """
        Converts a single index or a sequence of indices to texts.

        Args:
            ids (int|List[int]):
                The token id (or token ids) to be converted to text.

        Returns:
            str: The decoded text.

        Example:
            .. code-block::

                from paddlenlp.transformers import GPTTokenizer
                tokenizer = GPTTokenizer.from_pretrained('gpt2-medium-en')
                print(tokenizer.convert_ids_to_string(tokenizer.convert_ids_to_string([14618, 284, 779, 350, 37382, 47, 37382, 290, 350, 37382, 45, 19930]))
                # 'Welcome to use PaddlePaddle and PaddleNLP'

        """

        text = "".join([self.decoder[id] for id in ids])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            "utf-8", errors=self.errors
        )
        return text

    def save_resources(self, save_directory):
        """
        Saves `SentencePiece <https://github.com/google/sentencepiece>`__ file
        (ends with '.spm') under `save_directory`.

        Args:
            save_directory (str): Directory to save files into.
        """
        for name, file_name in self.resource_files_names.items():
            source_path = getattr(self, "_%s" % name)

            save_path = os.path.join(save_directory, file_name)
            if os.path.abspath(source_path) != os.path.abspath(save_path):
                shutil.copyfile(source_path, save_path)

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (string) in a single string.
        """
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            "utf-8", errors=self.errors
        )
        return text

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is None:
            return output

        return output + bos_token_ids + token_ids_1

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults

        # attention_mask shape [1,seq_len,seq_len]
        if (
            "attention_mask" in encoded_inputs
            and len(np.shape(encoded_inputs["attention_mask"])) > 2
        ):
            attention_mask = encoded_inputs["attention_mask"]
            encoded_inputs.pop("attention_mask")
        else:
            attention_mask = None

        required_input = encoded_inputs[self.model_input_names[0]]
        encoded_inputs = super()._pad(
            encoded_inputs,
            max_length,
            padding_strategy,
            pad_to_multiple_of,
            return_attention_mask,
        )
        if attention_mask is not None and len(np.shape(attention_mask)) > 2:
            encoded_inputs["attention_mask"] = attention_mask
            needs_to_be_padded = (
                padding_strategy != PaddingStrategy.DO_NOT_PAD
                and len(required_input) != max_length
            )
            if needs_to_be_padded:
                difference = max_length - len(required_input)
                if "attention_mask" in encoded_inputs:
                    encoded_inputs["attention_mask"] = np.pad(
                        encoded_inputs["attention_mask"],
                        pad_width=[(0, 0), (difference, 0), (difference, 0)],
                        mode="constant",
                        constant_values=0,
                    )
        return encoded_inputs
