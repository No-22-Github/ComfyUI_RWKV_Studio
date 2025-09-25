import os
import folder_paths

# 定义RWKV模型的存放路径
rwkv_models_path = os.path.join(folder_paths.models_dir, "RWKV")
supported_extensions = [".pth", ".pt", ".bin", ".safensors"]

# 确保文件夹存在
if not os.path.exists(rwkv_models_path):
    os.makedirs(rwkv_models_path)

class RWKVModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        # 扫描文件夹，获取模型文件列表
        model_files = [
            f for f in os.listdir(rwkv_models_path) 
            if os.path.isfile(os.path.join(rwkv_models_path, f)) and f.lower().endswith(tuple(supported_extensions))
        ]
        
        # 如果没有找到模型，提供一个提示
        if not model_files:
            model_files = ["No models found in ComfyUI/models/RWKV"]

        return {
            "required": {
                "model": (model_files, ),
            }
        }

    # 输出只有一个：模型路径字符串
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "load_model_info"
    CATEGORY = "RWKV_Studio"
    # 将分类统一为 RWKV_Studio

    def load_model_info(self, model):
        # 如果列表是提示信息，则返回空路径
        if model.startswith("No models found"):
            return ("",)

        # 构造并返回模型的完整路径
        model_path = os.path.join(rwkv_models_path, model)
        print(f"RWKV Loader: Selected model path: '{model_path}'")
        
        # 以元组形式返回
        return (model_path,)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "RWKV_ModelLoader_V7": RWKVModelLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RWKV_ModelLoader_V7": "RWKV Model Loader (V7 Only)"
}
