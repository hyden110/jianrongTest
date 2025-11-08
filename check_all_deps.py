#!/usr/bin/env python3
"""
全依赖检查脚本
验证所有依赖是否能够正常使用
"""

def check_package(package_name, import_name=None, version_attr=None):
    """检查包是否已安装并能正常导入"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        # 处理子模块的情况
        for sub_module in import_name.split('.')[1:]:
            module = getattr(module, sub_module)
        
        if version_attr and hasattr(module, version_attr):
            version = getattr(module, version_attr)
            print(f"✓ {package_name} (版本: {version})")
        else:
            print(f"✓ {package_name} (已安装)")
        return True
    except ImportError as e:
        print(f"✗ {package_name} (未安装或导入失败: {e})")
        return False
    except Exception as e:
        print(f"✗ {package_name} (错误: {e})")
        return False

def main():
    """主函数"""
    print("=== 全依赖检查 ===\n")
    
    # 需要检查的包列表
    packages = [
        # PyTorch生态
        ("torch", "torch", "__version__"),
        ("torchvision", "torchvision", "__version__"),
        ("torchaudio", "torchaudio", "__version__"),
        ("pytorch-lightning", "pytorch_lightning", "__version__"),
        ("fastai", "fastai", "__version__"),
        ("xformers", "xformers", None),
        ("flash_attn", "flash_attn", None),
        ("triton", "triton", "__version__"),
        ("cupy-cuda12x", "cupy", "__version__"),
        ("vllm", "vllm", None),
        
        # TensorFlow生态
        ("tensorflow", "tensorflow", "__version__"),
        ("keras", "keras", "__version__"),
        
        # PaddlePaddle生态
        ("paddlepaddle", "paddle", "__version__"),
        ("paddlepaddle-gpu", "paddle", "__version__"),
        ("paddledet", "ppdet", None),
        ("paddlex", "paddlex", None),
        ("paddleocr", "paddleocr", None),
        
        # OpenVINO
        ("openvino", "openvino", "__version__"),
        ("openvino-dev", "openvino", "__version__"),
        
        # TensorRT
        ("tensorrt", "tensorrt", "__version__"),
        ("tensorrt_cu13*", "tensorrt", "__version__"),
        
        # ONNX相关
        ("onnx", "onnx", "__version__"),
        ("onnxruntime", "onnxruntime", "__version__"),
        
        # 图像处理
        ("opencv-python", "cv2", "__version__"),
        ("opencv-contrib-python", "cv2", "__version__"),
        ("opencv-python-headless", "cv2", "__version__"),
        ("dlib", "dlib", "DLIB_VERSION"),
        
        # 数值计算
        ("numpy", "numpy", "__version__"),
        ("scipy", "scipy", "__version__"),
        ("pandas", "pandas", "__version__"),
        ("polars", "polars", "__version__"),
        ("scikit-learn", "sklearn", "__version__"),
        ("scikit-image", "skimage", "__version__"),
        ("ray", "ray", "__version__"),
        
        # 网络和序列化
        ("grpcio", "grpc", "__version__"),
        ("gradio", "gradio", "__version__"),
        ("streamlit", "streamlit", "__version__"),
        ("xformers", "xformers", None),
        ("tokenizers", "tokenizers", "__version__"),
        ("sentencepiece", "sentencepiece", "__version__"),
        
        # NVIDIA CUDA库
        ("nvidia-cuda-runtime-cu12", "nvidia.cuda_runtime", None),
        ("nvidia-cublas-cu12", "nvidia.cublas", None),
        ("nvidia-cudnn-cu12", "nvidia.cudnn", None),
        ("nvidia-nccl-cu12", "nvidia.nccl", None),
        ("nvidia-cuda-nvrtc-cu12", "nvidia.nvrtc", None),
        ("nvidia-cuda-cupti-cu12", "nvidia.cupti", None),
        ("nvidia-nvjitlink-cu12", "nvidia.nvjitlink", None),
        ("nvidia-cuda-runtime-cu13", "nvidia.cuda_runtime", None),
        ("nvidia-cublas-cu11", "nvidia.cublas", None),
        ("nvidia-cudnn-cu11", "nvidia.cudnn", None),
        ("nvidia-cuda-nvrtc-cu11", "nvidia.nvrtc", None),
    ]
    
    installed_count = 0
    total_count = len(packages)
    
    for package_name, import_name, version_attr in packages:
        if check_package(package_name, import_name, version_attr):
            installed_count += 1
    
    print(f"\n=== 检查结果 ===")
    print(f"已安装: {installed_count}/{total_count}")
    
    if installed_count == total_count:
        print("🎉 所有依赖都已安装!")
    else:
        print(f"⚠ {total_count - installed_count} 个依赖未安装或导入失败")

if __name__ == "__main__":
    main()