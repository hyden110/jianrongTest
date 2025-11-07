#!/usr/bin/env python3
"""
依赖验证脚本
验证以下依赖是否兼容并且能够使用：
torch, torchvision, xformers, flash_attn, vllm, transformers,
onnx, onnxruntime, opencv-python, opencv-contrib-python,
numpy, scipy, pandas, scikit-learn, grpcio, tokenizers, sentencepiece
以及相关的NVIDIA CUDA库
"""

import sys
import importlib
import traceback

def test_import(module_name, version_attr=None):
    """测试模块导入"""
    try:
        module = importlib.import_module(module_name)
        if version_attr and hasattr(module, version_attr):
            version = getattr(module, version_attr)
            print(f"✓ {module_name} (版本: {version})")
        else:
            print(f"✓ {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {module_name} (导入失败: {e})")
        return False
    except Exception as e:
        print(f"✗ {module_name} (错误: {e})")
        return False

def test_cuda_availability():
    """测试CUDA可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA可用 (GPU数量: {torch.cuda.device_count()})")
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
            return True
        else:
            print("⚠ CUDA不可用 (未检测到GPU或CUDA未正确安装)")
            return False
    except Exception as e:
        print(f"✗ CUDA测试失败: {e}")
        return False

def test_torch_functionality():
    """测试PyTorch基本功能"""
    try:
        import torch
        # 创建张量
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.mm(x, y)
        print(f"✓ PyTorch基本运算测试通过 (3x3矩阵乘法)")
        
        # 测试CUDA张量(如果可用)
        if torch.cuda.is_available():
            x_cuda = x.cuda()
            y_cuda = y.cuda()
            z_cuda = torch.mm(x_cuda, y_cuda)
            print(f"✓ PyTorch CUDA运算测试通过")
        
        return True
    except Exception as e:
        print(f"✗ PyTorch功能测试失败: {e}")
        return False

def test_transformers_functionality():
    """测试Transformers库功能"""
    try:
        import transformers
        # 检查版本
        print(f"✓ Transformers库导入成功 (版本: {transformers.__version__})")
        return True
    except Exception as e:
        print(f"✗ Transformers功能测试失败: {e}")
        return False

def test_opencv_functionality():
    """测试OpenCV功能"""
    try:
        import cv2
        import numpy as np
        
        # 创建一个简单的图像进行测试
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"✓ OpenCV基本功能测试通过")
        return True
    except Exception as e:
        print(f"✗ OpenCV功能测试失败: {e}")
        return False

def test_onnx_functionality():
    """测试ONNX功能"""
    try:
        import onnx
        print(f"✓ ONNX库导入成功 (版本: {onnx.__version__})")
        return True
    except Exception as e:
        print(f"✗ ONNX功能测试失败: {e}")
        return False

def test_pandas_functionality():
    """测试Pandas功能"""
    try:
        import pandas as pd
        import numpy as np
        
        # 创建简单的DataFrame
        df = pd.DataFrame({
            'A': np.random.randn(5),
            'B': np.random.randn(5)
        })
        print(f"✓ Pandas基本功能测试通过 (创建了{len(df)}行的DataFrame)")
        return True
    except Exception as e:
        print(f"✗ Pandas功能测试失败: {e}")
        return False

def test_numpy_functionality():
    """测试NumPy功能"""
    try:
        import numpy as np
        # 基本运算测试
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = np.dot(a, b)
        print(f"✓ NumPy基本功能测试通过 (点积结果: {c})")
        return True
    except Exception as e:
        print(f"✗ NumPy功能测试失败: {e}")
        return False

def test_scipy_functionality():
    """测试SciPy功能"""
    try:
        import scipy
        print(f"✓ SciPy库导入成功 (版本: {scipy.__version__})")
        return True
    except Exception as e:
        print(f"✗ SciPy功能测试失败: {e}")
        return False

def test_sklearn_functionality():
    """测试Scikit-learn功能"""
    try:
        import sklearn
        print(f"✓ Scikit-learn库导入成功 (版本: {sklearn.__version__})")
        return True
    except Exception as e:
        print(f"✗ Scikit-learn功能测试失败: {e}")
        return False

def test_grpcio_functionality():
    """测试gRPC功能"""
    try:
        import grpc
        print(f"✓ gRPC库导入成功 (版本: {grpc.__version__})")
        return True
    except Exception as e:
        print(f"✗ gRPC功能测试失败: {e}")
        return False

def test_tokenizers_functionality():
    """测试Tokenizers功能"""
    try:
        import tokenizers
        print(f"✓ Tokenizers库导入成功 (版本: {tokenizers.__version__})")
        return True
    except Exception as e:
        print(f"✗ Tokenizers功能测试失败: {e}")
        return False

def test_sentencepiece_functionality():
    """测试SentencePiece功能"""
    try:
        import sentencepiece
        print(f"✓ SentencePiece库导入成功 (版本: {sentencepiece.__version__})")
        return True
    except Exception as e:
        print(f"✗ SentencePiece功能测试失败: {e}")
        return False

def test_nvidia_cuda_functionality():
    """测试NVIDIA CUDA相关库功能"""
    nvidia_packages = [
        ("nvidia.cuda_runtime", "nvidia-cuda-runtime-cu12"),
        ("nvidia.cublas", "nvidia-cublas-cu12"),
        ("nvidia.cudnn", "nvidia-cudnn-cu12"),
        ("nvidia.nccl", "nvidia-nccl-cu12"),
        ("nvidia.nvrtc", "nvidia-cuda-nvrtc-cu12"),
        ("nvidia.cupti", "nvidia-cuda-cupti-cu12"),
        ("nvidia.nvjitlink", "nvidia-nvjitlink-cu12"),
        ("nvidia.cuda_runtime", "nvidia-cuda-runtime-cu13"),
        ("nvidia.cublas", "nvidia-cublas-cu11"),
        ("nvidia.cudnn", "nvidia-cudnn-cu11"),
        ("nvidia.nvrtc", "nvidia-cuda-nvrtc-cu11"),
    ]
    
    successful = 0
    for import_name, package_name in nvidia_packages:
        try:
            module = importlib.import_module(import_name)
            if hasattr(module, '__version__'):
                print(f"✓ {package_name} (版本: {module.__version__})")
            else:
                print(f"✓ {package_name}")
            successful += 1
        except ImportError:
            print(f"⚠ {package_name} 未安装")
        except Exception as e:
            print(f"✗ {package_name} 错误: {e}")
    
    print(f"NVIDIA CUDA库测试结果: {successful}/{len(nvidia_packages)} 成功")
    return successful

def main():
    """主函数"""
    print("=== Python依赖验证脚本 ===\n")
    
    # 获取Python版本
    print(f"Python版本: {sys.version}\n")
    
    # 要测试的模块列表
    modules_to_test = [
        # 深度学习相关
        ("torch", "__version__"),
        ("torchvision", "__version__"),
        ("transformers", "__version__"),
        ("tokenizers", "__version__"),
        ("sentencepiece", "__version__"),
        
        # 注意：flash_attn, vllm, xformers 可能需要特殊处理
        ("flash_attn", None),  # 特殊处理
        ("vllm", None),        # 特殊处理
        ("xformers", None),    # 特殊处理
        
        # ONNX相关
        ("onnx", "__version__"),
        ("onnxruntime", "__version__"),
        
        # 图像处理
        ("cv2", "__version__"),  # opencv-python
        # 注意：opencv-contrib-python通常与opencv-python一起安装
        
        # 数值计算
        ("numpy", "__version__"),
        ("scipy", "__version__"),
        ("pandas", "__version__"),
        ("sklearn", "__version__"),  # scikit-learn
        
        # 网络和序列化
        ("grpcio", "__version__"),
    ]
    
    # 特殊处理的模块
    special_modules = [
        "flash_attn",
        "vllm",
        "xformers"
    ]
    
    print("1. 测试基本模块导入...")
    successful_imports = 0
    total_imports = len(modules_to_test)
    
    for module_name, version_attr in modules_to_test:
        if test_import(module_name, version_attr):
            successful_imports += 1
    
    print(f"\n模块导入测试结果: {successful_imports}/{total_imports} 成功\n")
    
    print("2. 测试特殊模块...")
    for module_name in special_modules:
        try:
            # 特殊处理这些模块
            if module_name == "flash_attn":
                import flash_attn
                print(f"✓ {module_name}")
            elif module_name == "vllm":
                import vllm
                print(f"✓ {module_name}")
            elif module_name == "xformers":
                import xformers
                print(f"✓ {module_name}")
        except ImportError:
            print(f"⚠ {module_name} 未安装或导入失败 (这在某些环境中是正常的)")
        except Exception as e:
            print(f"✗ {module_name} 错误: {e}")
    
    print("\n3. 测试功能...")
    
    # 测试CUDA
    print("\n[CUDA测试]")
    test_cuda_availability()
    
    # 测试各库功能
    print("\n[功能测试]")
    test_torch_functionality()
    test_transformers_functionality()
    test_opencv_functionality()
    test_onnx_functionality()
    test_pandas_functionality()
    test_numpy_functionality()
    test_scipy_functionality()
    test_sklearn_functionality()
    test_grpcio_functionality()
    test_tokenizers_functionality()
    test_sentencepiece_functionality()
    
    # 测试NVIDIA CUDA相关库
    print("\n[NVIDIA CUDA库测试]")
    test_nvidia_cuda_functionality()
    
    print("\n=== 测试完成 ===")
    
    if successful_imports == total_imports:
        print("🎉 所有模块都成功导入!")
    else:
        print(f"⚠ {total_imports - successful_imports} 个模块导入失败，请检查依赖安装")
        print("\n建议:")
        print("1. 对于torchvision问题，可能是Python安装缺少_lzma模块")
        print("2. 对于flash_attn, vllm, xformers，可能需要特殊安装步骤或不兼容当前环境")
        print("3. 可以尝试使用conda安装这些包以获得更好的兼容性")

if __name__ == "__main__":
    main()