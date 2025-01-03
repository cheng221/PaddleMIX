import io
import re


from PIL import Image
import paddle
import paddle.vision.transforms as T

from .constants import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD, IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN, SIGLIP_MEAN, SIGLIP_STD

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