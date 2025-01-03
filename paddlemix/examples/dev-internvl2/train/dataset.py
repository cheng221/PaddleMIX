import os
import io
import random
import re
from collections import Counter
from typing import Dict,TYPE_CHECKING

import paddle
import cv2
import imageio
import numpy as np
from decord import VideoReader
from PIL import Image

from .constants import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD, IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN, SIGLIP_MEAN, SIGLIP_STD
from .trainer_utils import LabelSmoother
from .conversation_utils import get_conv_template
# from .vision_utils import build_transform

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
except ImportError as E:
    print(
        'petrel_client is not installed. If you read data locally instead of from ceph, ignore it.'
        )
import sys

if TYPE_CHECKING:
    from paddlenlp.transformers import PreTrainedTokenizer

def calculate_ngram_repetition(text, n):
    words = text.split()
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    ngram_counts = Counter(ngrams)
    total_ngrams = len(ngrams)
    repeated_ngrams = sum(1 for count in ngram_counts.values() if count > 1)
    return repeated_ngrams / total_ngrams if total_ngrams > 0 else 0


def check_conversations_repetition(conversations, repeat_threshold=0.4,
    ngram=10):
    for conversation in conversations:
        if conversation['from'] == 'gpt':
            model_answer = conversation['value']
            repeat_ratio = calculate_ngram_repetition(model_answer, ngram)
            if repeat_ratio > repeat_threshold:
                raise Exception


def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None,
    input_fps=1, max_num_frames=-1):
    if sample in ['rand', 'middle']:
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1
            ).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in
                    ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                paddle.sort(x=frame_indices), paddle.argsort(x=frame_indices)
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [(x[0] + fix_start) for x in ranges]
        elif sample == 'middle':
            frame_indices = [((x[0] + x[1]) // 2) for x in ranges]
        else:
            raise NotImplementedError
        if len(frame_indices) < num_frames:
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif 'fps' in sample:
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
    else:
        raise ValueError
    return frame_indices


def read_frames_gif(video_path, num_frames, sample='rand', fix_start=None,
    client=None, min_num_frames=4):
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        gif = imageio.get_reader(io.BytesIO(video_bytes))
    else:
        gif = imageio.get_reader(video_path)
    vlen = len(gif)
    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample,
        fix_start=fix_start)
    frames = []
    for index, frame in enumerate(gif):
        if index in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB).astype(np.uint8)
            frame = Image.fromarray(frame)
            frames.append(frame)
    return frames


def read_frames_decord(video_path, num_frames, sample='rand', fix_start=
    None, client=None, clip=None, min_num_frames=4):
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)
    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample,
        fix_start=fix_start, input_fps=fps)
    if clip:
        frame_indices = [(f + start_index) for f in frame_indices]
    frames = video_reader.get_batch(frame_indices).asnumpy()
    frames = [Image.fromarray(frames[i]) for i in range(tuple(frames.shape)[0])
        ]
    return frames


def extract_frame_number(filename):
    match = re.search('_(\\d+).jpg$', filename)
    return int(match.group(1)) if match else -1


def sort_frames(frame_paths):
    return sorted(frame_paths, key=lambda x: extract_frame_number(os.path.
        basename(x)))


def read_frames_folder(video_path, num_frames, sample='rand', fix_start=
    None, client=None, clip=None, min_num_frames=4):
    if 's3://' in video_path:
        image_list = sort_frames(client.list(video_path))
        frames = []
        for image in image_list:
            fp = os.path.join(video_path, image)
            frame = Image.open(io.BytesIO(client.get(fp)))
            frames.append(frame)
    else:
        image_list = sort_frames(list(os.listdir(video_path)))
        frames = []
        for image in image_list:
            fp = os.path.join(video_path, image)
            frame = Image.open(fp).convert('RGB')
            frames.append(frame)
    vlen = len(frames)
    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    if vlen > t_num_frames:
        frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample,
            fix_start=fix_start)
        frames = [frames[i] for i in frame_indices]
    return frames


class WeightedConcatDataset(paddle.io.ConcatDataset):

    def __init__(self, datasets, weights):
        super().__init__(datasets)
        self.weights = paddle.to_tensor(data=weights, dtype='float64')
        self.total_size = sum(len(d) for d in datasets)
        self.sampler = paddle.io.WeightedRandomSampler(weights=self.weights,
            num_samples=self.total_size, replacement=True)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return self.total_size


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    return img.convert('RGB')


class TCSLoader(object):

    def __init__(self, conf_path, sc_config_key='sensecore'):
        print(f'[TCSLoader] config_path: {conf_path}')
        print('--> before Client(conf_path)')
        self.client = Client(conf_path)
        self.sc_config_key = sc_config_key
        print('--> after Client(conf_path)')

    def __call__(self, fn, image_type='image', max_num_frames=-1,
        min_num_frames=8, sample='rand', clip=None):
        if image_type == 'image':
            img_value_str = self.client.get(fn)
            img = pil_loader(img_value_str)
            return img
        elif image_type == 'video':
            if fn.endswith('/'):
                frames = read_frames_folder(fn, num_frames=max_num_frames,
                    min_num_frames=min_num_frames, client=self.client,
                    sample=sample)
            elif fn.endswith('.gif'):
                frames = read_frames_gif(fn, num_frames=max_num_frames,
                    min_num_frames=min_num_frames, client=self.client,
                    sample=sample)
            else:
                frames = read_frames_decord(fn, num_frames=max_num_frames,
                    min_num_frames=min_num_frames, client=self.client,
                    sample=sample, clip=clip)
            return frames

def preprocess(template_name, sources, tokenizer: "PreTrainedTokenizer", num_image_token_list: list, text_only: bool=False,
    group_by_length: bool=False, use_packed_ds: bool=False, ds_name: str=
    None, num_image: int=1) ->Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())
    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = (
                    f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                    )
                conversation = conversation.replace('<image>', image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations
    input_ids = tokenizer(conversations, return_tensors='pt', padding=False if
        group_by_length or use_packed_ds else 'max_length', max_length=
        tokenizer.model_max_length, truncation=True).input_ids
    targets = input_ids.clone()
    sep = conv.sep + conv.roles[1] + ': '
    for conversation, target in zip(conversations, targets):
        total_len = int(target.not_equal(y=paddle.to_tensor(tokenizer.
            pad_token_id)).sum())
        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == '':
                break
            turn_len = len(tokenizer(turn).input_ids)
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            if i != 0 and not tokenizer.legacy:
                instruction_len -= 1
            target[cur_len:cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len
            if i != 0 and not tokenizer.legacy:
                cur_len -= 1
        target[cur_len:] = IGNORE_TOKEN_ID
        if False:
            z = target.clone()
            z = paddle.where(condition=z == IGNORE_TOKEN_ID, x=tokenizer.
                unk_token_id, y=z)
            logger.info(tokenizer.decode(z))
            exit()
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}. #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                    )
                sys.stdout.flush()
    return dict(input_ids=input_ids, labels=targets, attention_mask=
        input_ids.not_equal(y=paddle.to_tensor(tokenizer.pad_token_id)))


def preprocess_mpt(template_name, sources, tokenizer: paddlenlp.
    transformers.PretrainedTokenizer, num_image_token_list: list, text_only:
    bool=False, group_by_length: bool=False, use_packed_ds: bool=False,
    ds_name: str=None, num_image: int=1) ->Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())
    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = (
                    f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                    )
                conversation = conversation.replace('<image>', image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations
    input_ids = tokenizer(conversations, return_tensors='pt', padding=False if
        group_by_length or use_packed_ds else 'max_length', max_length=
        tokenizer.model_max_length, truncation=True).input_ids
    targets = input_ids.clone()
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.not_equal(y=paddle.to_tensor(tokenizer.
            pad_token_id)).sum())
        turns = conversation.split(conv.sep)
        re_turns = [conv.sep.join(turns[:3])]
        for conv_idx in range(3, len(turns), 2):
            re_turns.append(conv.sep.join(turns[conv_idx:conv_idx + 2]))
        cur_len = 0
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(re_turns):
            if turn == '':
                break
            turn_len = len(tokenizer(turn).input_ids) + 1
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            instruction_len = len(tokenizer(parts[0]).input_ids)
            target[cur_len:cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len
        target[cur_len:] = IGNORE_TOKEN_ID
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}. #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                    )
                sys.stdout.flush()
    return dict(input_ids=input_ids, labels=targets, attention_mask=
        input_ids.not_equal(y=paddle.to_tensor(tokenizer.pad_token_id)))


def preprocess_phi3(template_name, sources, tokenizer: paddlenlp.
    transformers.PretrainedTokenizer, num_image_token_list: list, text_only:
    bool=False, group_by_length: bool=False, use_packed_ds: bool=False,
    ds_name: str=None, num_image: int=1) ->Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())
    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = (
                    f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                    )
                conversation = conversation.replace('<image>', image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations
    tokenizer.padding_side = 'right'
    input_ids = tokenizer(conversations, return_tensors='pt', padding=False if
        group_by_length or use_packed_ds else 'max_length', max_length=
        tokenizer.model_max_length, truncation=True).input_ids
    targets = input_ids.clone()
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.not_equal(y=paddle.to_tensor(int(tokenizer.
            pad_token_id))).sum())
        turns = conversation.split(conv.sep)
        re_turns = [conv.sep.join(turns[:3])]
        for conv_idx in range(3, len(turns), 2):
            re_turns.append(conv.sep.join(turns[conv_idx:conv_idx + 2]))
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        endoftext_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        target[target == endoftext_id] = IGNORE_TOKEN_ID
        for i, turn in enumerate(re_turns):
            if turn == '':
                break
            if i == 0:
                turn_len = len(tokenizer(turn).input_ids)
            else:
                turn_len = len(tokenizer(turn).input_ids) - 1
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            if i == 0:
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
            else:
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            target[cur_len:cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len
        target[cur_len:] = IGNORE_TOKEN_ID
        if False:
            z = target.clone()
            z = paddle.where(condition=z == IGNORE_TOKEN_ID, x=tokenizer.
                unk_token_id, y=z)
            print(repr(tokenizer.decode(z)))
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}. #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                    )
                sys.stdout.flush()
    return dict(input_ids=input_ids, labels=targets, attention_mask=
        input_ids.not_equal(y=paddle.to_tensor(tokenizer.pad_token_id)))


def preprocess_internlm(template_name, sources, tokenizer: paddlenlp.
    transformers.PretrainedTokenizer, num_image_token_list: list, text_only:
    bool=False, group_by_length: bool=False, use_packed_ds: bool=False,
    ds_name: str=None, num_image: int=1) ->Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            sentence['value'] = sentence['value'].strip()
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())
    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = (
                    f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                    )
                conversation = conversation.replace('<image>', image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations
    input_ids = tokenizer(conversations, return_tensors='pt', padding=False if
        group_by_length or use_packed_ds else 'max_length', max_length=
        tokenizer.model_max_length, truncation=True).input_ids
    targets = input_ids.clone()
    for conversation, target in zip(conversations, targets):
        total_len = int(target.not_equal(y=paddle.to_tensor(tokenizer.
            pad_token_id)).sum())
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        parts = conversation.split(conv.roles[1])
        info = parts[0] + conv.roles[1]
        temp_len = len(tokenizer(info).input_ids) - 1
        target[cur_len:cur_len + temp_len] = IGNORE_TOKEN_ID
        cur_len = cur_len + temp_len
        for index in range(1, len(parts) - 1):
            info = parts[index]
            part1, part2 = info.split(conv.roles[0])
            temp_len = len(tokenizer(part1).input_ids) - 1
            cur_len = cur_len + temp_len
            part = conv.roles[0] + part2 + conv.roles[1]
            temp_len = len(tokenizer(part).input_ids) - 1
            target[cur_len:cur_len + temp_len] = IGNORE_TOKEN_ID
            cur_len = cur_len + temp_len
        last_info = parts[-1]
        temp_len = len(tokenizer(last_info).input_ids) - 1
        cur_len = cur_len + temp_len
        target[cur_len:] = IGNORE_TOKEN_ID
        if False:
            z = target.clone()
            z = paddle.where(condition=z == IGNORE_TOKEN_ID, x=tokenizer.
                unk_token_id, y=z)
            print(repr(tokenizer.decode(z)))
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}. This dataset is {ds_name}.'
                    )
                sys.stdout.flush()
    return dict(input_ids=input_ids, labels=targets, attention_mask=
        input_ids.not_equal(y=paddle.to_tensor(tokenizer.pad_token_id)))


def preprocess_internvl2_5(template_name, sources, tokenizer: paddlenlp.
    transformers.PretrainedTokenizer, num_image_token_list: list, text_only:
    bool=False, group_by_length: bool=False, use_packed_ds: bool=False,
    ds_name: str=None, num_image: int=1) ->Dict:
    assert len(sources) == 1, 'process only the first conversations'
    conversations = sources[0]
    if conversations[0]['from'] == 'system':
        system_prompt = conversations[0]['value']
        conversations = conversations[1:]
    else:
        conv = get_conv_template(template_name)
        system_prompt = conv.system_message
    if not text_only:
        new_conversations = []
        current_image_idx = 0
        for conversation in conversations:
            if conversation['from'] == 'human':
                image_cnt = conversation['value'].count('<image>')
                for i in range(image_cnt):
                    if current_image_idx == num_image:
                        break
                    image_tokens = (
                        f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[current_image_idx]}{IMG_END_TOKEN}'
                        )
                    conversation['value'] = conversation['value'].replace(
                        '<image>', image_tokens, 1)
                    current_image_idx += 1
            new_conversations.append(conversation)
        conversations = new_conversations
        assert current_image_idx == num_image, f'{current_image_idx} != {num_image}'
    batches, roles = [], []
    if system_prompt is not None:
        batches.append(f'<|im_start|>system\n{system_prompt}<|im_end|>\n')
        roles.append('system')
    for conversation in conversations:
        if conversation['from'] == 'human':
            batches.append(
                f"<|im_start|>user\n{conversation['value']}<|im_end|>\n")
            roles.append('human')
        elif conversation['from'] == 'gpt':
            batches.append(
                f"<|im_start|>assistant\n{conversation['value']}<|im_end|>\n")
            roles.append('gpt')
        else:
            raise NotImplementedError
    if tokenizer.add_bos_token:
        batches[0] = tokenizer.bos_token + batches[0]
    input_ids = tokenizer(batches, return_tensors='np', padding=False,
        max_length=tokenizer.model_max_length, truncation=False).input_ids
    if tokenizer.add_bos_token:
        input_ids = [item[1:] for item in input_ids]
    final_input_ids, final_targets = [], []
    ignore_ids = tokenizer('<|im_start|>assistant\n', return_tensors='np'
        ).input_ids[0]
    ignore_len = tuple(ignore_ids.shape)[0
        ] - 1 if tokenizer.add_bos_token else tuple(ignore_ids.shape)[0]
    for role, input_id in zip(roles, input_ids):
        final_input_ids.append(input_id)
        if role == 'system' or role == 'human':
            final_targets.append(np.full(tuple(input_id.shape),
                IGNORE_TOKEN_ID))
        elif role == 'gpt':
            target = input_id.copy()
            target[:ignore_len] = IGNORE_TOKEN_ID
            target[-1:] = IGNORE_TOKEN_ID
            final_targets.append(target)
        else:
            raise NotImplementedError
    input_ids = paddle.to_tensor(data=np.concatenate(final_input_ids))[:
        tokenizer.model_max_length]
    targets = paddle.to_tensor(data=np.concatenate(final_targets))[:
        tokenizer.model_max_length]
    padding = False if group_by_length or use_packed_ds else True
    if padding:
        current_length = input_ids.shape[0]
        padding_length = tokenizer.model_max_length - current_length
        input_ids = paddle.nn.functional.pad(x=input_ids, pad=(0,
            padding_length), value=tokenizer.pad_token_id,
            pad_from_left_axis=False)
        targets = paddle.nn.functional.pad(x=targets, pad=(0,
            padding_length), value=IGNORE_TOKEN_ID, pad_from_left_axis=False)
    input_ids = input_ids.unsqueeze(axis=0)
    targets = targets.unsqueeze(axis=0)
    return dict(input_ids=input_ids, labels=targets, attention_mask=
        input_ids.not_equal(y=paddle.to_tensor(tokenizer.pad_token_id)))


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
    image_size):
    best_ratio_diff = float('inf')
    best_ratio = 1, 1
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

