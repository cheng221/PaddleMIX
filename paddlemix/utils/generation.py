# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
""" Logits Processor Helper class for Emu3. """
from typing import List, Union

import paddle
from paddlenlp.generation import StoppingCriteria


class EosTokenCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the "end-of-sequence" token is generated.
    By default, it uses the `model.generation_config.eos_token_id`.

    Args:
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
    """

    def __init__(self, eos_token_id: Union[int, List[int], paddle.Tensor]):
        if not isinstance(eos_token_id, paddle.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = paddle.to_tensor(eos_token_id)
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor, **kwargs) -> paddle.Tensor:
        self.eos_token_id = self.eos_token_id.to(input_ids.place)
        is_done = paddle.isin(input_ids[:, -1], self.eos_token_id)
        return is_done
