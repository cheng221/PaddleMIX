from paddlenlp.transformers import Qwen2Tokenizer
from paddlemix.models.internvl2 import InternVLChatConfig
from paddlemix.models.internvl2.internvl_chat import InternVLChatModel

def load_model_tokenizer(pretrained_path,dtype="bfloat16"):
    kwargs = {} 
    tokenizer = Qwen2Tokenizer.from_pretrained(pretrained_path, use_fast=False,trust_remote_code=True)
    model = InternVLChatModel.from_pretrained(pretrained_path, dtype=dtype).eval()
    return model, tokenizer