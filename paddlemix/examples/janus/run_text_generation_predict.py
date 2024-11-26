import paddle
import os
from paddlemix.auto import AutoModelMIX
from paddlemix.processors import JanusVLChatProcessor,JanusImageProcessor
from paddlenlp.transformers import LlamaTokenizerFast

import numpy as np
import PIL.Image
from utils.io import load_pil_images
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="deepseek-ai/Janus-1.3B")
parser.add_argument("--image_file", type=str, required=True)
args = parser.parse_args()

vl_gpt = AutoModelMIX.from_pretrained(args.model_path)
tokenizer = LlamaTokenizerFast.from_pretrained(args.model_path)
image_processer = JanusImageProcessor.from_pretrained(args.model_path)
vl_chat_processor: JanusVLChatProcessor = JanusVLChatProcessor(image_processer,tokenizer)

conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>\nWhat is shown in this image?",
        "images": [args.image_file],
    },
    {"role": "Assistant", "content": ""},
]

pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(conversations=conversation, images=
    pil_images, force_batchify=True)

inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

bs, seq_len = prepare_inputs.attention_mask.shape
position_ids = paddle.arange(seq_len, dtype=paddle.int64).reshape([1, -1])

outputs = vl_gpt.language_model.generate(
    input_ids=prepare_inputs['input_ids'],
    inputs_embeds=inputs_embeds,
    position_ids=position_ids,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=128, #512,
    do_sample=False,
    use_cache=True,
)
answer = tokenizer.decode(outputs[0][0], skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)