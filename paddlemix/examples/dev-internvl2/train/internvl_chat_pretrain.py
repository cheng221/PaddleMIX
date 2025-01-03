import os
import paddle
import paddlenlp
import json
import logging
import math
import random
import sys
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Literal, Optional
import numpy as np
from internvl.dist_utils import init_dist
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.openelm.modeling_openelm import OpenELMForCausalLM
from internvl.model.internvl_chat import InternVisionConfig, InternVisionModel, InternVLChatConfig, InternVLChatModel
from internvl.patch import concat_pad_data_collator, replace_internlm2_attention_class, replace_llama_attention_class, replace_llama_rmsnorm_with_fused_rmsnorm, replace_phi3_attention_class, replace_qwen2_attention_class, replace_train_dataloader, replace_train_sampler
from internvl.train.constants import BOX_END_TOKEN, BOX_START_TOKEN, IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN, QUAD_END_TOKEN, QUAD_START_TOKEN, REF_END_TOKEN, REF_START_TOKEN
from internvl.train.dataset import ConcatDataset, TCSLoader, WeightedConcatDataset, build_transform, check_conversations_repetition, dynamic_preprocess, preprocess, preprocess_internlm, preprocess_internvl2_5, preprocess_mpt, preprocess_phi3
from internvl.train.dataset_packed import PackedDataset, packed_collate_fn
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
replace_llama_rmsnorm_with_fused_rmsnorm()
replace_train_sampler()
replace_train_dataloader()
try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(default=None, metadata={
        'help':
        'Path to a pretrained model (local or from huggingface.co/models).'})
    vision_path: Optional[str] = field(default=None, metadata={'help':
        'Path to a pretrained model (local or from huggingface.co/models).'})
    llm_path: Optional[str] = field(default=None, metadata={'help':
        'Path to a pretrained model (local or from huggingface.co/models).'})
    mlp_path: Optional[str] = field(default=None, metadata={'help':
        'Path to a pretrained model (local or from huggingface.co/models).'})
    freeze_llm: bool = field(default=False, metadata={'help':
        'Set to True to freeze the LLM. Default is False.'})
    freeze_backbone: bool = field(default=False, metadata={'help':
        'Set to True to freeze the ViT. Default is False.'})
    freeze_mlp: bool = field(default=False, metadata={'help':
        'Set to True to freeze the MLP. Default is False.'})
    unfreeze_vit_layers: int = field(default=0, metadata={'help':
        'Specify the number of ViT layers to unfreeze. Default is 0.'})
    vision_select_layer: int = field(default=-1, metadata={'help':
        'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'
        })
    use_backbone_lora: int = field(default=0, metadata={'help':
        'Set the LoRA adapter rank for the ViT. Default is 0.'})
    use_llm_lora: int = field(default=0, metadata={'help':
        'Set the LoRA adapter rank for the LLM. Default is 0.'})
    unfreeze_lm_head: bool = field(default=False, metadata={'help':
        'Set to True to unfreeze the head of LLM. Default is False.'})
    grad_checkpoint: bool = field(default=True, metadata={'help':
        'Set to True to use gradient checkpointing. Default is True.'})
    drop_path_rate: float = field(default=0.0, metadata={'help':
        'Set the drop path rate for the ViT. Default is 0.'})
    ps_version: Literal['v1', 'v2'] = field(default='v2', metadata={'help':
        'Specify the version of pixel shuffle implementation. Default is v2.'})
    use_fast_tokenizer: bool = field(default=False, metadata={'help':
        'Set to True to use the fast mode of the tokenizer.'})


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    max_seq_length: int = field(default=8192, metadata={'help':
        'The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.'
        })
    force_image_size: int = field(default=448, metadata={'help':
        'Set the desired size for the image. Default is 448.'})
    down_sample_ratio: float = field(default=0.5, metadata={'help':
        'Set the desired down-sampling ratio for the image. Default is 0.5.'})
    pad2square: bool = field(default=False, metadata={'help':
        'Pad the image to a square shape if set to True. Default is False.'})
    conv_style: str = field(default='internlm2-chat', metadata={'help':
        'Prompt style for a conversation.'})
    meta_path: str = field(default=None, metadata={'help':
        'The path of the meta file of datasets.'})
    use_data_resampling: bool = field(default=False, metadata={'help':
        'Set to True to use data resampling. Default is False.'})
    dynamic_image_size: bool = field(default=False, metadata={'help':
        'Set to True to use dynamic high resolution strategy. Default is False.'
        })
    use_thumbnail: bool = field(default=False, metadata={'help':
        'Set to True to add a thumbnail image. Default is False.'})
    min_dynamic_patch: Optional[int] = field(default=1, metadata={'help':
        'The minimum number of dynamic patches. Default is 1.'})
    max_dynamic_patch: Optional[int] = field(default=12, metadata={'help':
        'The maximum number of dynamic patches. Default is 12.'})
    normalize_type: Literal['imagenet', 'clip', 'siglip'] = field(default=
        'imagenet', metadata={'help':
        'The normalization type for the image. Default is imagenet.'})
    use_packed_ds: bool = field(default=False, metadata={'help':
        'Whether to use packed dataset for efficient training. Default is False.'
        })
    num_images_expected: int = field(default=40, metadata={'help':
        'The maximum number of images per packed sample. Default is 40.'})
    max_packed_tokens: int = field(default=8192, metadata={'help':
        'The required token length of per packed sample. Default is 8192.'})
    max_buffer_size: int = field(default=20, metadata={'help':
        'The buffer size of the packed dataset. Default is 20.'})
    log_freq: int = field(default=1000, metadata={'help':
        'The log frequency of the packed dataset. Default is 1000.'})
    strict_mode: bool = field(default=True, metadata={'help':
        'Whether to pad the number of images to satisfy num_images_expected. Default is True.'
        })
    replacement: bool = field(default=False, metadata={'help':
        'Whether to restart the dataset after it is exhausted. Default is False.'
        })
    allow_overflow: bool = field(default=False, metadata={'help':
        'Whether to drop the sample over the specified max_packed_tokens. Default is False.'
        })
    loss_reduction: str = field(default='token', metadata={'help':
        'Loss reduction method. Default is token.'})
    loss_reduction_all_gather: bool = field(default=False, metadata={'help':
        'Whether to gather all during loss reduction. Default is False.'})


class LazySupervisedDataset(paddle.io.Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, template_name, meta, tokenizer, tcs_loader, ds_name,
        num_image_token, image_size=448, is_train=True, pad2square=False,
        group_by_length=False, dynamic_image_size=False, use_thumbnail=
        False, min_dynamic_patch=1, max_dynamic_patch=12, min_num_frame=8,
        max_num_frame=32, sampling_method='rand', repeat_time=1,
        normalize_type='imagenet', use_packed_ds=False, data_rank=0,
        data_world_size=1, distributed_mode=False, force_shuffle=False,
        random_seed=0):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(
            f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}'
            )
        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode
        self.dataset_type = 'pair'
        self.max_num_images = 1
        self.max_tokens = tokenizer.model_max_length
        self.force_shuffle = force_shuffle
        self._state_dict = {}
        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'
            ), f"annotation must be jsonl, but got {meta['annotation']}"
>>>>>>        total_ranks = torch.distributed.get_world_size()
        self.total_ranks = total_ranks
        current_rank = paddle.distributed.get_rank()
        """
        This section of the code is used to read hundreds of millions of data entries.
        By using caching and splitting the data according to rank, it ensures fast reading
        speed and prevents out-of-memory.
        """
        basename = os.path.basename(meta['annotation']).replace('.jsonl', '')
        data_dir = os.path.join(os.path.dirname(meta['annotation']),
            f'{basename}_temp')
        os.makedirs(data_dir, exist_ok=True)
        temp_path = os.path.join(data_dir,
            f'{basename}_{current_rank}_of_{total_ranks}.jsonl')
        if os.path.exists(temp_path):
            with open(temp_path, 'r') as f:
                self.raw_data = f.readlines()
        else:
            with open(meta['annotation'], 'r') as f:
                self.raw_data = f.readlines()
            if repeat_time < 1:
                self.raw_data = self.raw_data[:int(len(self.raw_data) *
                    repeat_time)]
            else:
                self.raw_data = self.raw_data * int(repeat_time)
            total_lines = len(self.raw_data)
            logger.info(
                f'total_ranks: {total_ranks}, current_rank: {current_rank}, total_lines: {total_lines}'
                )
            lines_per_rank = total_lines // total_ranks
            lines_per_rank = max(1, lines_per_rank)
            start_line = lines_per_rank * current_rank
            end_line = start_line + lines_per_rank
            self.raw_data = self.raw_data[start_line:end_line]
            with open(temp_path, 'w') as f:
                f.writelines(self.raw_data)
        self.rng = np.random.default_rng(seed=random_seed)
        if self.force_shuffle:
            self.rng.shuffle(self.raw_data)
        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type
        assert not group_by_length
        if self.group_by_length:
            self.conv2length = {}
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']
                else:
                    conversations = '\n'.join([temp['value'] for temp in
                        data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(conversations,
                            return_tensors='pt', padding=False, truncation=
                            False).input_ids.size(1)
                        self.conv2length[str_length
                            ] = token_length + num_image_token * (
                            max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

    def __len__(self):
        if not self.use_packed_ds:
            return len(self.raw_data) * self.total_ranks
        else:
            return len(self.raw_data)

    def get_preprocess_function(self):
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        elif self.template_name == 'internvl2_5':
            preprocess_function = preprocess_internvl2_5
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):
            image_path = self.root + image_path
        else:
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        transform = build_transform(is_train=self.is_train, input_size=self
            .image_size, pad2square=self.pad2square, normalize_type=self.
            normalize_type)
        return transform

    def multi_modal_get_item(self, data_item):
        transform = self.get_transform()
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item[
                'conversations'][0]['value']
        image_path = self.get_image_path(data_item['image'])
        image = self.load_image(image_path)
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, min_num=self.
                min_dynamic_patch, max_num=self.max_dynamic_patch,
                image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:
            images = [image]
        pixel_values = [transform(image) for image in images]
        pixel_values = paddle.stack(x=pixel_values)
        num_patches = pixel_values.shape[0]
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
        preprocess_function = self.get_preprocess_function()
        ret = preprocess_function(self.template_name, [deepcopy(data_item[
            'conversations'])], self.tokenizer, [self.num_image_token *
            num_patches], group_by_length=self.group_by_length,
            use_packed_ds=self.use_packed_ds, ds_name=self.ds_name)
        position_ids = ret['attention_mask'].astype(dtype='int64').cumsum(axis
            =-1) - 1
        position_ids.masked_fill_(mask=ret['attention_mask'] == 0, value=1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN
            )
        assert (ret['input_ids'][0] == image_end_token_id).sum(
            ) == 1, f'image tokens are truncated, this dataset is {self.ds_name}'
        ret = dict(input_ids=ret['input_ids'][0], labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0], position_ids=
            position_ids[0], pixel_values=pixel_values, image_flags=paddle.
            to_tensor(data=[1] * num_patches, dtype='int64'))
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        transform = self.get_transform()
        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            image_path = self.get_image_path(image_path)
            image = self.load_image(image_path)
            if self.dynamic_image_size:
                image = dynamic_preprocess(image, min_num=self.
                    min_dynamic_patch, max_num=max(1, self.
                    max_dynamic_patch // num_image), image_size=self.
                    image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = paddle.stack(x=pixel_values)
        num_patches = pixel_values.shape[0]
        preprocess_function = self.get_preprocess_function()
        num_image_tokens = [(self.num_image_token * num_tile) for num_tile in
            num_tiles]
        ret = preprocess_function(self.template_name, [deepcopy(data_item[
            'conversations'])], self.tokenizer, num_image_tokens,
            group_by_length=self.group_by_length, use_packed_ds=self.
            use_packed_ds, ds_name=self.ds_name, num_image=num_image)
        position_ids = ret['attention_mask'].astype(dtype='int64').cumsum(axis
            =-1) - 1
        position_ids.masked_fill_(mask=ret['attention_mask'] == 0, value=1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN
            )
        assert (ret['input_ids'][0] == image_end_token_id).sum(
            ) == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'
        ret = dict(input_ids=ret['input_ids'][0], labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0], position_ids=
            position_ids[0], pixel_values=pixel_values, image_flags=paddle.
            to_tensor(data=[1] * num_patches, dtype='int64'))
        return ret

    def video_get_item(self, data_item):
        transform = self.get_transform()
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<video>\n' + data_item[
                'conversations'][0]['value']
        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)
        image_list = self.tcs_loader(video_path, image_type='video',
            max_num_frames=self.max_num_frame, min_num_frames=self.
            min_num_frame, sample=self.sampling_method, clip=data_item.get(
            'clip', None))
        special_tokens = '\n'.join(['Frame-{}: <image>'.format(i + 1) for i in
            range(len(image_list))])
        data_item['conversations'][0]['value'] = data_item['conversations'][0][
            'value'].replace('<video>\n', special_tokens + '\n')
        pixel_values = [transform(image) for image in image_list]
        pixel_values = paddle.stack(x=pixel_values)
        num_patches = pixel_values.shape[0]
        preprocess_function = self.get_preprocess_function()
        num_image_tokens = [self.num_image_token] * num_patches
        ret = preprocess_function(self.template_name, [deepcopy(data_item[
            'conversations'])], self.tokenizer, num_image_tokens,
            group_by_length=self.group_by_length, use_packed_ds=self.
            use_packed_ds, ds_name=self.ds_name, num_image=num_patches)
        position_ids = ret['attention_mask'].astype(dtype='int64').cumsum(axis
            =-1) - 1
        position_ids.masked_fill_(mask=ret['attention_mask'] == 0, value=1)
        ret = dict(input_ids=ret['input_ids'][0], labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0], position_ids=
            position_ids[0], pixel_values=pixel_values, image_flags=paddle.
            to_tensor(data=[1] * num_patches, dtype='int64'))
        return ret

    def pure_text_get_item(self, data_item):
        transform = self.get_transform()
        image = Image.new('RGB', (224, 224), (255, 255, 255))
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
            max_num=1, image_size=self.image_size, use_thumbnail=self.
            use_thumbnail)
        pixel_values = [transform(image) for image in images]
        pixel_values = paddle.stack(x=pixel_values)
        num_patches = pixel_values.shape[0]
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
        preprocess_function = self.get_preprocess_function()
        ret = preprocess_function(self.template_name, [deepcopy(data_item[
            'conversations'])], self.tokenizer, [self.num_image_token *
            num_patches], text_only=True, group_by_length=self.
            group_by_length, use_packed_ds=self.use_packed_ds, ds_name=self
            .ds_name)
        position_ids = ret['attention_mask'].astype(dtype='int64').cumsum(axis
            =-1) - 1
        position_ids.masked_fill_(mask=ret['attention_mask'] == 0, value=1)
        ret = dict(input_ids=ret['input_ids'][0], labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0], position_ids=
            position_ids[0], pixel_values=pixel_values, image_flags=paddle.
            to_tensor(data=[0] * num_patches, dtype='int64'))
        return ret

    def _enable_worker_distributed(self):
        if (self.distributed_mode and not self.worker_distributed and self.
            worker_id is not None):
            self.worker_distributed = True
            num_worker_per_rank = self.num_workers // self.total_ranks
            self.raw_data = self.raw_data[self.worker_id %
                num_worker_per_rank::num_worker_per_rank]
            logger.info(
                f'worker_distributed is enabled, self.num_workers={self.num_workers!r}, len(self.raw_data)={len(self.raw_data)!r}'
                )

    def __getitem__(self, i) ->Dict[str, paddle.Tensor]:
        if i >= len(self.raw_data):
            if self.use_packed_ds:
                raise NotImplementedError
            else:
                i = i % len(self.raw_data)
        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'
                    ] is not None and data_item['video'] != '':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                print(e, self.ds_name, flush=True)
                if not isinstance(e, UnidentifiedImageError):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [(self.root + item) for item in data_item[
                            'image']]
                        print(
                            f'Failed to load image: {images}, the dataset is: {self.ds_name}'
                            )
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item[
                                'image'])
                        print(
                            f'Failed to load image: {data_path}, the dataset is: {self.ds_name}'
                            )
                elif 'video' in data_item:
                    data_path = os.path.join(self.root, data_item['video'])
                    print(
                        f'Failed to load video: {data_path}, the dataset is: {self.ds_name}'
                        )
                i = random.randint(0, len(self.raw_data) - 1)
        return ret

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0
        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self.
            _state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']
            self._state_dict.pop(self.worker_state_key)
        if self.worker_id == 0:
            logger.info(
                f'[{self.ds_name}] [Worker id {self.worker_id}] begin to iter with start_idx={start_idx!r}'
                )
        for i in range(start_idx, len(self)):
            yield self[i]


def build_datasets(data_args, tokenizer, tcs_loader, model, group_by_length
    =False, dynamic_image_size=False, use_thumbnail=False,
    min_dynamic_patch=1, max_dynamic_patch=12, normalize_type='imagenet'):
    datasets = []
    lengths = []
    data_rank = paddle.distributed.get_rank()
>>>>>>    data_world_size = torch.distributed.get_world_size()
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(
                f'max_dynamic_patch is set to {max_num} according to the meta file'
                )
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(data_args.conv_style,
            ds_collections[ds_name], tokenizer, tcs_loader, ds_name=ds_name,
            num_image_token=model.num_image_token, image_size=data_args.
            force_image_size, is_train=ds_collections[ds_name][
            'data_augment'], pad2square=data_args.pad2square,
            group_by_length=group_by_length and not data_args.use_packed_ds,
            dynamic_image_size=dynamic_image_size, use_thumbnail=
            use_thumbnail, min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num, repeat_time=repeat_time,
            normalize_type=normalize_type, use_packed_ds=data_args.
            use_packed_ds, data_rank=data_rank, data_world_size=
            data_world_size, distributed_mode=data_args.use_packed_ds,
            force_shuffle=data_args.use_packed_ds, random_seed=ds_idx)
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))
    if data_args.use_packed_ds:
        total_length = sum(lengths)
        train_dataset = PackedDataset(tokenizer=tokenizer, data_rank=
            data_rank, data_world_size=data_world_size, datasets=datasets,
            dataset_weight=[(l / total_length) for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens, max_buffer_size=
            data_args.max_buffer_size, log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode, replacement=data_args.
            replacement, allow_overflow=data_args.allow_overflow,
            allow_deduplicated_ds_name=False)
    elif data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [(l / total_length) for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / x ** 0.5
    raise NotImplementedError(loss_reduction)


def main():
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
>>>>>>    parser = transformers.HfArgumentParser((ModelArguments,
        DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        model_args, data_args, training_args = parser.parse_json_file(json_file
            =os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = (parser.
            parse_args_into_dataclasses())
    training_args.use_packed_ds = data_args.use_packed_ds
    logging.basicConfig(format=
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt=
        '%m/%d/%Y %H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
    if training_args.should_log:
>>>>>>        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
>>>>>>    transformers.utils.logging.set_verbosity(log_level)
>>>>>>    transformers.utils.logging.enable_default_handler()
>>>>>>    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.place}, n_gpu: {training_args.n_gpu}'
         +
        f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
        )
    logger.info(f'Training/evaluation parameters {training_args}')
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir
        ) and training_args.do_train and not training_args.overwrite_output_dir:
>>>>>>        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(
            training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)
            ) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.'
                )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
                )
>>>>>>    transformers.set_seed(training_args.seed)
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
>>>>>>    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path,
        add_eos_token=False, trust_remote_code=True, use_fast=model_args.
        use_fast_tokenizer)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
        QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN, REF_END_TOKEN,
        BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    tcs_loader = TCSLoader('~/petreloss.conf') if has_tcs_loader else None
    if data_args.use_packed_ds:
        replace_internlm2_attention_class()
        replace_qwen2_attention_class()
        replace_phi3_attention_class()
        replace_llama_attention_class()
    if model_args.model_name_or_path is not None:
        logger.info('Loading InternVLChatModel...')
        config = InternVLChatConfig.from_pretrained(model_args.
            model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'
            logger.info('Using flash_attention_2 for InternLM')
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'
            logger.info('Using flash_attention_2 for LLaMA')
        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        model = InternVLChatModel.from_pretrained(model_args.
            model_name_or_path, torch_dtype='bfloat16', config=config)
    else:
        logger.info('Loading ViT-6B...')
        vision_config = InternVisionConfig.from_pretrained(model_args.
            vision_path)
        vision_config.drop_path_rate = model_args.drop_path_rate
        vision_model = InternVisionModel.from_pretrained(model_args.
            vision_path, torch_dtype='bfloat16', config=vision_config)
        logger.info('Loading LLaMA...')
>>>>>>        llm_config = transformers.AutoConfig.from_pretrained(model_args.
            llm_path, trust_remote_code=True)
        if llm_config.model_type == 'internlm2':
            model_type = InternLM2ForCausalLM
            llm_config.attn_implementation = 'flash_attention_2'
            logger.info('Using flash_attention_2 for InternLM')
        else:
>>>>>>            model_type = transformers.AutoModelForCausalLM
            llm_config._attn_implementation = 'flash_attention_2'
            logger.info('Using flash_attention_2 for LLaMA')
        llm = model_type.from_pretrained(model_args.llm_path, torch_dtype=
            'bfloat16', config=llm_config, trust_remote_code=True)
        logger.info('Building InternVLChatConfig...')
        internvl_chat_config = InternVLChatConfig(vision_config.to_dict(),
            llm_config.to_dict(), downsample_ratio=data_args.
            down_sample_ratio, pad2square=data_args.pad2square, template=
            data_args.conv_style, select_layer=model_args.
            vision_select_layer, dynamic_image_size=data_args.
            dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
            ps_version=model_args.ps_version, min_dynamic_patch=data_args.
            min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch)
        internvl_chat_config.force_image_size = data_args.force_image_size
        logger.info('Building InternVLChatModel...')
        model = InternVLChatModel(internvl_chat_config, vision_model, llm)
    model.img_context_token_id = img_context_token_id
    assert model.config.downsample_ratio == data_args.down_sample_ratio
    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = paddle.load(path=str(model_args.mlp_path))
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info('Finished')
    patch_size = model.config.vision_config.patch_size
    logger.info(
        f'model.config.force_image_size: {model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(
        f'model.config.vision_config.image_size: {model.config.vision_config.image_size}'
        )
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(
            f'Resizing position embedding from {model.config.vision_config.image_size} to {data_args.force_image_size}...'
            )
        model.vision_model.resize_pos_embeddings(old_size=model.config.
            vision_config.image_size, new_size=data_args.force_image_size,
            patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) **
        2 * data_args.down_sample_ratio ** 2)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings(
            ).weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(axis
            =0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)
    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()
    train_dataset = build_datasets(data_args, tokenizer, tcs_loader, model,
        group_by_length=training_args.group_by_length, dynamic_image_size=
        data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=
        data_args.max_dynamic_patch, normalize_type=data_args.normalize_type)

    def _freeze_params(module):
        for param in module.parameters():
            param.stop_gradient = not False
    if model_args.freeze_backbone:
        _freeze_params(model.vision_model)
    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)
    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.stop_gradient = not True
    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha
            =2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora
    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 *
            model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora
    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)
    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.
            unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.stop_gradient = not True
    if paddle.distributed.get_rank() == 0:
        for name, param in model.named_parameters():
            if not param.stop_gradient:
                logger.info(name)
>>>>>>    transformers.set_seed(training_args.seed)
    if data_args.use_packed_ds:
        collator = partial(packed_collate_fn, data_collator=
            concat_pad_data_collator, max_item_length=data_args.
            max_packed_tokens if data_args.strict_mode else 0, micro_num=
            training_args.train_batch_size, len2weight=partial(len2weight,
            loss_reduction=data_args.loss_reduction),
            loss_reduction_all_gather=data_args.loss_reduction_all_gather)
    else:
        collator = concat_pad_data_collator
>>>>>>    trainer = transformers.Trainer(model=model, args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None, tokenizer=tokenizer, data_collator=collator)
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()


if __name__ == '__main__':
    main()
