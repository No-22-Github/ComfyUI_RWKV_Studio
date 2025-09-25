import os
import importlib
import inspect

os.environ["RWKV_JIT_ON"] = '0'

os.environ["RWKV_CUDA_ON"] = '0'


# 获取当前 __init__.py 文件所在的目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 初始化最终要提供给 ComfyUI 的两个核心字典
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 遍历当前目录下的所有文件
for filename in os.listdir(current_dir):
    # 确保文件是 Python 文件 (.py) 并且不是自己 (__init__.py)
    if filename.endswith(".py") and filename != "__init__.py":
        
        # 从文件名中获取模块名 (例如 "RWKV_Loader.py" -> "RWKV_Loader")
        module_name = filename[:-3]
        
        try:
            # 动态导入模块。 
            # f".{module_name}" 表示从当前包（由__name__指定）中进行相对导入
            module = importlib.import_module(f".{module_name}", __name__)

            # 检查导入的模块中是否有名为 NODE_CLASS_MAPPINGS 的变量
            if hasattr(module, "NODE_CLASS_MAPPINGS") and isinstance(module.NODE_CLASS_MAPPINGS, dict):
                # 如果有，就将其内容合并到主字典中
                NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                print(f"✅ Loaded nodes from {filename}")

            # 同样检查 NODE_DISPLAY_NAME_MAPPINGS
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") and isinstance(module.NODE_DISPLAY_NAME_MAPPINGS, dict):
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

        except Exception as e:
            # 如果导入或处理过程中出现任何错误，打印出来，方便排查
            print(f"❌ Failed to load nodes from {filename}: {e}")


# 这是Python包的标准做法，告诉ComfyUI要导出哪些变量
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("---")
print("RWKV_Studio Auto-Load:")
print(f"   - Found {len(NODE_CLASS_MAPPINGS)} node classes.")
print(f"   - Found {len(NODE_DISPLAY_NAME_MAPPINGS)} display names.")
print("---")
