import os
import gc
import torch

os.environ['RWKV_CUDA_ON'] = '0'
os.environ['RWKV_V7_ON'] = '1'

try:
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE, PIPELINE_ARGS
except ImportError as e:
    print(f"Error: RWKV Translator node is missing the 'rwkv' library. Please run: python -m pip install rwkv in your ComfyUI's Python environment")
    raise ImportError(f"RWKV Translator could not be loaded, please install the 'rwkv' library.") from e

# --- 全局配置和模型缓存 ---
_model_cache = {}
CTX_LIMIT = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_PATH = "rwkv_vocab_v20230424" 

# --- 核心生成函数 ---
def _rwkv_evaluate(model, pipeline, ctx):
    args = PIPELINE_ARGS(
        temperature=1.0, top_p=0.7,
        alpha_frequency=0.1, alpha_presence=0.1,
        token_ban=[], token_stop=[0]
    )
    ctx, all_tokens, out_last, out_str, occurrence, state = ctx.strip(), [], 0, '', {}, None
    for i in range(CTX_LIMIT):
        input_ids = pipeline.encode(ctx)[-CTX_LIMIT:] if i == 0 else [token]
        out, state = model.forward(input_ids, state)
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)
        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
            break
        all_tokens.append(token)
        for xxx in occurrence:
            occurrence[xxx] *= 0.996
        decoded_token = pipeline.decode([token])
        is_common = decoded_token in ' \t0123456789'
        penalty_weight = 0 if is_common else 1
        if token not in occurrence:
            occurrence[token] = penalty_weight
        else:
            occurrence[token] += penalty_weight
        tmp = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:
            out_str += tmp
            out_last = i + 1
    del out, state, occurrence, all_tokens
    gc.collect()
    if 'cuda' in str(DEVICE):
        torch.cuda.empty_cache()
    return out_str.strip()

def perform_translation(model_path, direction, text_to_translate):
    if not text_to_translate.strip():
        return ""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: '{model_path}'")

    if model_path in _model_cache:
        model, pipeline = _model_cache[model_path]
        print(f"RWKV Translator: Fetching model '{os.path.basename(model_path)}' from cache")
    else:
        print(f"RWKV Translator: Loading new model from '{model_path}'...")
        try:
            model_base_path, _ = os.path.splitext(model_path)
            model = RWKV(model=model_base_path, strategy=f'{DEVICE} fp16')
            pipeline = PIPELINE(model, VOCAB_PATH)
            _model_cache[model_path] = (model, pipeline)
            print(f"RWKV Translator: Model loaded successfully to {DEVICE} and cached.")
        except Exception as e:
            if model_path in _model_cache:
                del _model_cache[model_path]
            raise RuntimeError(f"Failed to load RWKV model: {e}")

    prompt_template = "English: {text}\n\nChinese:" if direction == 'en2zh' else "Chinese: {text}\n\nEnglish:"
    full_prompt = prompt_template.format(text=text_to_translate)
    
    return _rwkv_evaluate(model, pipeline, full_prompt)

class RWKVTranslator:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
                "model_path": ("STRING", {"forceInput": True}),
                "direction": (["en2zh", "zh2en"],),
                "text_to_translate": ("STRING", {"multiline": True, "default": "Welcome to the RWKV series models, embrace the RNN architecture."}),
            }}
            
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)
    FUNCTION = "execute_translation"
    CATEGORY = "RWKV_Studio"

    def execute_translation(self, model_path, direction, text_to_translate):
        translated_text = ""
        try:
            translated_text = perform_translation(model_path, direction, text_to_translate)
            print("RWKV Translator: Translation complete.")
        except Exception as e:
            print(f"Error: RWKV Translator execution failed: {e}")
            translated_text = f"Translation failed: {e}"
            
        return (translated_text,)

NODE_CLASS_MAPPINGS = { "RWKV_Translator_Node": RWKVTranslator }
NODE_DISPLAY_NAME_MAPPINGS = { "RWKV_Translator_Node": "RWKV Translator" }