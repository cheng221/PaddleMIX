# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import re

import paddle
from paddlenlp.generation import StoppingCriteriaList
from PIL import Image
from train.load_utils import load_model_tokenizer

from paddlemix.datasets.internvl_dataset import build_transform, dynamic_preprocess
from paddlemix.utils.generation import EosTokenCriteria
from paddlemix.utils.tools import check_dtype_compatibility

paddle.set_grad_enabled(False)


def post_processing(response):
    response = response.replace("\n", "").replace("不是", "No").replace("是", "Yes").replace("否", "No")
    response = response.lower().replace("true", "yes").replace("false", "no")
    pattern = re.compile(r"[\u4e00-\u9fa5]")
    response = re.sub(pattern, "", response)
    return response


def load_image(image_file, input_size=224, max_num=1, use_thumbnail=False, dynamic=False):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(is_train=False, input_size=input_size)
    if dynamic:
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num)
    else:
        images = [image]
    pixel_values = [transform(image) for image in images]
    pixel_values = paddle.stack(pixel_values)
    return pixel_values


def main(args):
    model, tokenizer = load_model_tokenizer(args.model_name_or_path)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    if args.image_path is not None and args.image_path != "None":
        pixel_values = load_image(args.image_path, image_size, args.max_num, use_thumbnail, args.dynamic).to(
            args.dtype
        )
        args.text = "<image>\n" + args.text

    else:
        pixel_values = None
    stopping_criteria = StoppingCriteriaList(
        [
            EosTokenCriteria(int(tokenizer.eos_token_id)),
        ]
    )
    generation_config = dict(
        do_sample=args.sample,
        top_k=args.top_k,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=20,
        eos_token_id=tokenizer.eos_token_id,
        trunc_input=False,
        stopping_criteria=stopping_criteria,
    )

    with paddle.no_grad():
        response, history = model.chat(
            tokenizer, pixel_values, args.text, generation_config, history=None, return_history=True
        )
        print(f"User: {args.text}\nAssistant: {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="OpenGVLab/InternVL2-8B",
        help="pretrained ckpt and tokenizer",
    )
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--text", type=str, default="Please describe the image shortly.", required=True)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"], help="Model dtype"
    )
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sample", type=bool, default=False)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--max-num", type=int, default=6)
    args = parser.parse_args()

    if args.dtype == "bfloat16":
        args.dtype = paddle.bfloat16
    elif args.dtype == "float16":
        args.dtype = paddle.float16
    else:
        args.dtype = paddle.float32

    # 检查环境支持的dtype并设置
    available_dtype = check_dtype_compatibility()

    # 如果用户指定了dtype，尝试使用用户指定的类型
    if args.dtype == "bfloat16":
        desired_dtype = paddle.bfloat16
    elif args.dtype == "float16":
        desired_dtype = paddle.float16
    else:
        desired_dtype = paddle.float32

    # 如果用户指定的dtype不可用，使用检测到的可用dtype
    if desired_dtype != available_dtype:
        print(f"Warning: Requested dtype {args.dtype} is not available, using {available_dtype}")
        args.dtype = available_dtype
    else:
        args.dtype = desired_dtype

    print(f"Using dtype: {args.dtype}")
    main(args)
