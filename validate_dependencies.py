#!/usr/bin/env python3
"""
依赖验证脚本
验证以下依赖是否兼容并且能够使用：
torch, torchvision, torchaudio, pytorch-lightning, fastai, xformers, flash_attn, triton, cupy-cuda12x, vllm,
transformers, onnx, onnxruntime, tensorflow, keras, paddlepaddle, paddlepaddle-gpu, paddledet, paddlex, paddleocr,
openvino, openvino-dev, tensorrt, tensorrt_cu13*,
nvidia-cuda-runtime-cu12, nvidia-cublas-cu12, nvidia-cudnn-cu12, nvidia-nccl-cu12, nvidia-cuda-nvrtc-cu12, 
nvidia-cuda-cupti-cu12, nvidia-nvjitlink-cu12, nvidia-cuda-runtime-cu13, nvidia-cublas-cu11, nvidia-cudnn-cu11, 
nvidia-cuda-nvrtc-cu11,
opencv-python, opencv-contrib-python, opencv-python-headless, dlib,
numpy, scipy, pandas, polars, scikit-learn, scikit-image, ray,
grpcio, gradio, streamlit, xformers, tokenizers, sentencepiece
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

def test_tensorflow_functionality():
    """测试TensorFlow功能"""
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow库导入成功 (版本: {tf.__version__})")
        return True
    except ImportError:
        print("⚠ TensorFlow未安装")
        return False
    except Exception as e:
        print(f"✗ TensorFlow功能测试失败: {e}")
        return False

def test_keras_functionality():
    """测试Keras功能"""
    try:
        import keras
        print(f"✓ Keras库导入成功 (版本: {keras.__version__})")
        return True
    except ImportError:
        print("⚠ Keras未安装")
        return False
    except Exception as e:
        print(f"✗ Keras功能测试失败: {e}")
        return False

def test_paddle_functionality():
    """测试PaddlePaddle功能"""
    try:
        import paddle
        print(f"✓ PaddlePaddle库导入成功 (版本: {paddle.__version__})")
        return True
    except ImportError:
        print("⚠ PaddlePaddle未安装")
        return False
    except Exception as e:
        print(f"✗ PaddlePaddle功能测试失败: {e}")
        return False

def test_openvino_functionality():
    """测试OpenVINO功能"""
    try:
        import openvino
        print(f"✓ OpenVINO库导入成功 (版本: {openvino.__version__})")
        return True
    except ImportError:
        print("⚠ OpenVINO未安装")
        return False
    except Exception as e:
        print(f"✗ OpenVINO功能测试失败: {e}")
        return False

def test_tensorrt_functionality():
    """测试TensorRT功能"""
    try:
        import tensorrt
        print(f"✓ TensorRT库导入成功 (版本: {tensorrt.__version__})")
        return True
    except ImportError:
        print("⚠ TensorRT未安装")
        return False
    except Exception as e:
        print(f"✗ TensorRT功能测试失败: {e}")
        return False

def test_dlib_functionality():
    """测试Dlib功能"""
    try:
        import dlib
        print(f"✓ Dlib库导入成功 (版本: {dlib.DLIB_VERSION})")
        return True
    except ImportError:
        print("⚠ Dlib未安装")
        return False
    except Exception as e:
        print(f"✗ Dlib功能测试失败: {e}")
        return False

def test_polars_functionality():
    """测试Polars功能"""
    try:
        import polars as pl
        print(f"✓ Polars库导入成功 (版本: {pl.__version__})")
        return True
    except ImportError:
        print("⚠ Polars未安装")
        return False
    except Exception as e:
        print(f"✗ Polars功能测试失败: {e}")
        return False

def test_skimage_functionality():
    """测试Scikit-image功能"""
    try:
        import skimage
        print(f"✓ Scikit-image库导入成功 (版本: {skimage.__version__})")
        return True
    except ImportError:
        print("⚠ Scikit-image未安装")
        return False
    except Exception as e:
        print(f"✗ Scikit-image功能测试失败: {e}")
        return False

def test_ray_functionality():
    """测试Ray功能"""
    try:
        import ray
        print(f"✓ Ray库导入成功 (版本: {ray.__version__})")
        return True
    except ImportError:
        print("⚠ Ray未安装")
        return False
    except Exception as e:
        print(f"✗ Ray功能测试失败: {e}")
        return False

def test_gradio_functionality():
    """测试Gradio功能"""
    try:
        import gradio
        print(f"✓ Gradio库导入成功 (版本: {gradio.__version__})")
        return True
    except ImportError:
        print("⚠ Gradio未安装")
        return False
    except Exception as e:
        print(f"✗ Gradio功能测试失败: {e}")
        return False

def test_streamlit_functionality():
    """测试Streamlit功能"""
    try:
        import streamlit
        print(f"✓ Streamlit库导入成功 (版本: {streamlit.__version__})")
        return True
    except ImportError:
        print("⚠ Streamlit未安装")
        return False
    except Exception as e:
        print(f"✗ Streamlit功能测试失败: {e}")
        return False

def test_pytorch_lightning_functionality():
    """测试PyTorch Lightning功能"""
    try:
        import pytorch_lightning
        print(f"✓ PyTorch Lightning库导入成功 (版本: {pytorch_lightning.__version__})")
        return True
    except ImportError:
        print("⚠ PyTorch Lightning未安装")
        return False
    except Exception as e:
        print(f"✗ PyTorch Lightning功能测试失败: {e}")
        return False

def test_fastai_functionality():
    """测试FastAI功能"""
    try:
        import fastai
        print(f"✓ FastAI库导入成功 (版本: {fastai.__version__})")
        return True
    except ImportError:
        print("⚠ FastAI未安装")
        return False
    except Exception as e:
        print(f"✗ FastAI功能测试失败: {e}")
        return False

def test_torchaudio_functionality():
    """测试Torchaudio功能"""
    try:
        import torchaudio
        print(f"✓ Torchaudio库导入成功 (版本: {torchaudio.__version__})")
        return True
    except ImportError:
        print("⚠ Torchaudio未安装")
        return False
    except Exception as e:
        print(f"✗ Torchaudio功能测试失败: {e}")
        return False

def test_triton_functionality():
    """测试Triton功能"""
    try:
        import triton
        print(f"✓ Triton库导入成功 (版本: {triton.__version__})")
        return True
    except ImportError:
        print("⚠ Triton未安装")
        return False
    except Exception as e:
        print(f"✗ Triton功能测试失败: {e}")
        return False

def test_cupy_functionality():
    """测试CuPy功能"""
    try:
        import cupy
        print(f"✓ CuPy库导入成功 (版本: {cupy.__version__})")
        return True
    except ImportError:
        print("⚠ CuPy未安装")
        return False
    except Exception as e:
        print(f"✗ CuPy功能测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=== Python依赖验证脚本 ===\n")
    
    # 获取Python版本
    print(f"Python版本: {sys.version}\n")
    
    # 要测试的模块列表
    modules_to_test = [
        # 深度学习框架
        ("torch", "__version__"),
        ("torchvision", "__version__"),
        ("torchaudio", "__version__"),
        ("pytorch_lightning", "__version__"),
        ("fastai", "__version__"),
        ("transformers", "__version__"),
        ("xformers", None),
        ("flash_attn", None),
        ("triton", "__version__"),
        ("cupy", "__version__"),
        ("vllm", None),
        
        # TensorFlow生态
        ("tensorflow", "__version__"),
        ("keras", "__version__"),
        
        # PaddlePaddle生态
        ("paddle", "__version__"),
        
        # OpenVINO
        ("openvino", "__version__"),
        
        # TensorRT
        ("tensorrt", "__version__"),
        
        # ONNX相关
        ("onnx", "__version__"),
        ("onnxruntime", "__version__"),
        
        # 图像处理
        ("cv2", "__version__"),  # opencv-python
        ("dlib", "DLIB_VERSION"),
        
        # 数值计算
        ("numpy", "__version__"),
        ("scipy", "__version__"),
        ("pandas", "__version__"),
        ("polars", "__version__"),
        ("sklearn", "__version__"),  # scikit-learn
        ("skimage", "__version__"),  # scikit-image
        
        # 分布式计算
        ("ray", "__version__"),
        
        # 网络和序列化
        ("grpcio", "__version__"),
        ("gradio", "__version__"),
        ("streamlit", "__version__"),
        ("tokenizers", "__version__"),
        ("sentencepiece", "__version__"),
    ]
    
    # 特殊处理的模块
    special_modules = [
        "xformers",
        "flash_attn",
        "vllm"
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
            if module_name == "xformers":
                import xformers
                print(f"✓ {module_name}")
            elif module_name == "flash_attn":
                import flash_attn
                print(f"✓ {module_name}")
            elif module_name == "vllm":
                import vllm
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
    test_torchaudio_functionality()
    test_transformers_functionality()
    test_pytorch_lightning_functionality()
    test_fastai_functionality()
    test_triton_functionality()
    test_cupy_functionality()
    test_opencv_functionality()
    test_dlib_functionality()
    test_onnx_functionality()
    test_pandas_functionality()
    test_numpy_functionality()
    test_scipy_functionality()
    test_sklearn_functionality()
    test_polars_functionality()
    test_skimage_functionality()
    test_ray_functionality()
    test_grpcio_functionality()
    test_tokenizers_functionality()
    test_sentencepiece_functionality()
    test_tensorflow_functionality()
    test_keras_functionality()
    test_paddle_functionality()
    test_openvino_functionality()
    test_tensorrt_functionality()
    test_gradio_functionality()
    test_streamlit_functionality()
    
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