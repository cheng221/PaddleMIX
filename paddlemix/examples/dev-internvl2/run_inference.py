import argparse
import io
import re

import paddle
import paddle.vision.transforms as T
from PIL import Image
from paddlenlp.generation import (
    GenerationConfig,
    MaxLengthCriteria,
    StoppingCriteriaList,
)
from paddlemix.datasets.internvl_dataset import dynamic_preprocess
from paddlemix.utils.generation import EosTokenCriteria

from utils import load_model_tokenizer

paddle.set_grad_enabled(False)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def post_processing(response):
    response = response.replace('\n', '').replace('不是', 'No').replace('是', 'Yes').replace('否', 'No')
    response = response.lower().replace('true', 'yes').replace('false', 'no')
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    response = re.sub(pattern, '', response)
    return response

 
def simulate_jpeg_degradation(quality):
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert('RGB').save(output, format='JPEG', quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg
    return jpeg_degrade

qualities = list(range(75, 101))
jpeg_degrade_functions = {quality: simulate_jpeg_degradation(quality) for quality in qualities}

class AdaptiveTransform(paddle.vision.transforms.BaseTransform):
    def __init__(self, keys):
        self.keys = [keys]
    
    def __call__(self,inputs):
        if not isinstance(inputs, tuple):
            inputs = (inputs,)

        self.params = self._get_params(inputs)
        outputs = []
        for i in range(min(len(inputs), len(self.keys))):
            apply_func = self._get_apply(self.keys[i])
            if apply_func is None:
                outputs.append(inputs[i])
            else:
                outputs.append(apply_func(inputs[i]))
        if len(inputs) > len(self.keys):
            outputs.extend(inputs[len(self.keys) :])

        if len(outputs) == 1:
            outputs = outputs[0]
        else:
            outputs = tuple(outputs)
        return outputs

def check_dtype_compatibility():
    """
    检查当前环境下可用的数据类型
    返回最优的可用数据类型
    """
    if not paddle.is_compiled_with_cuda():
        print("CUDA not available, falling back to float32")
        return paddle.float32

    # 获取GPU计算能力
    gpu_arch = paddle.device.cuda.get_device_capability()
    if gpu_arch is None:
        print("Unable to determine GPU architecture, falling back to float32")
        return paddle.float32
    
    major, minor = gpu_arch
    compute_capability = major + minor/10
    print(f"GPU compute capability: {compute_capability}")
    
    try:
        # 测试bfloat16兼容性
        if compute_capability >= 8.0:  # Ampere及更新架构
            test_tensor = paddle.zeros([2, 2], dtype='bfloat16')
            test_op = paddle.matmul(test_tensor, test_tensor)
            print("bfloat16 is supported and working")
            return paddle.bfloat16
    except Exception as e:
        print(f"bfloat16 test failed: {str(e)}")

    try:
        # 测试float16兼容性
        if compute_capability >= 5.3:  # Maxwell及更新架构
            test_tensor = paddle.zeros([2, 2], dtype='float16')
            test_op = paddle.matmul(test_tensor, test_tensor)
            print("float16 is supported and working")
            return paddle.float16
    except Exception as e:
        print(f"float16 test failed: {str(e)}")

    print("Falling back to float32 due to compatibility issues")
    return paddle.float32

def build_transform(is_train, input_size, pad2square=False, normalize_type='imagenet'):
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError
    if is_train:  # use data augumentation
        transform = T.Compose([
            AdaptiveTransform(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomChoice([AdaptiveTransform(jpeg_degrade_functions[quality]) for quality in qualities]),
            T.Resize((input_size, input_size), interpolation="bicubic"),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        if pad2square is False:  # now we use this transform function by default
            transform = T.Compose([
                AdaptiveTransform(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation="bicubic"),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
        else:
            transform = T.Compose([
                AdaptiveTransform(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                AdaptiveTransform(lambda img: expand2square(img, tuple(int(x * 255) for x in MEAN))),
                T.Resize((input_size, input_size), interpolation="bicubic"),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])

    return transform

def load_image(image_file, input_size=224,max_num=1,use_thumbnail=False,dynamic=False):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(is_train=False, input_size=input_size)
    if dynamic:
        images = dynamic_preprocess(image, image_size=input_size,
                                    use_thumbnail=use_thumbnail,
                                    max_num=max_num)
    else:
        images = [image]
    pixel_values = [transform(image) for image in images]
    pixel_values = paddle.stack(pixel_values)
    return pixel_values

def main(args):
    model,tokenizer = load_model_tokenizer(args.model_name_or_path)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    if args.image_path is not None and args.image_path != "None":
        pixel_values = load_image(args.image_path,image_size,args.max_num,use_thumbnail,args.dynamic).to(args.dtype)
        args.text = "<image>\n" + args.text

    else:
        pixel_values = None
    generation_config = dict(do_sample=args.sample, top_k=args.
        top_k, top_p=args.top_p, num_beams=args.num_beams,
        max_new_tokens=20, eos_token_id=tokenizer.eos_token_id,trunc_input=False)
    input_query,template,history = model.prepare_query(
        tokenizer,pixel_values,question=args.text,
    )
    input_ids = tokenizer(input_query)
    stopping_criteria = StoppingCriteriaList(
        [
            MaxLengthCriteria(generation_config['max_new_tokens']+len(input_ids)),
            EosTokenCriteria(int(tokenizer.eos_token_id)),
        ]
    )

    generation_config['stopping_criteria'] = stopping_criteria
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
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Model dtype"
    )
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
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