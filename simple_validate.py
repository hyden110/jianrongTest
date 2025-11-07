#!/usr/bin/env python3
"""
简化版依赖验证脚本
验证已成功安装的核心依赖是否能够正常使用
"""

def check_torch():
    """检查PyTorch"""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        # 基本运算测试
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        z = torch.mm(x, y)
        print(f"  基本运算测试: 通过")
        
        # CUDA测试
        if torch.cuda.is_available():
            print(f"  CUDA可用: 是 (GPU数量: {torch.cuda.device_count()})")
        else:
            print(f"  CUDA可用: 否")
        return True
    except Exception as e:
        print(f"✗ PyTorch: {e}")
        return False

def check_transformers():
    """检查Transformers"""
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        return True
    except Exception as e:
        print(f"✗ Transformers: {e}")
        return False

def check_opencv():
    """检查OpenCV"""
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
        
        # 基本功能测试
        import numpy as np
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"  基本功能测试: 通过")
        return True
    except Exception as e:
        print(f"✗ OpenCV: {e}")
        return False

def check_numpy():
    """检查NumPy"""
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        # 基本运算测试
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = np.dot(a, b)
        print(f"  基本运算测试: 通过 (点积结果: {c})")
        return True
    except Exception as e:
        print(f"✗ NumPy: {e}")
        return False

def check_pandas():
    """检查Pandas"""
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
        
        # 基本功能测试
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        print(f"  基本功能测试: 通过 (DataFrame形状: {df.shape})")
        return True
    except Exception as e:
        print(f"✗ Pandas: {e}")
        return False

def check_onnx():
    """检查ONNX"""
    try:
        import onnx
        print(f"✓ ONNX {onnx.__version__}")
        return True
    except Exception as e:
        print(f"✗ ONNX: {e}")
        return False

def check_onnxruntime():
    """检查ONNX Runtime"""
    try:
        import onnxruntime
        print(f"✓ ONNX Runtime {onnxruntime.__version__}")
        return True
    except Exception as e:
        print(f"✗ ONNX Runtime: {e}")
        return False

def check_grpc():
    """检查gRPC"""
    try:
        import grpc
        print(f"✓ gRPC {grpc.__version__}")
        return True
    except Exception as e:
        print(f"✗ gRPC: {e}")
        return False

def main():
    """主函数"""
    print("=== 简化版依赖验证 ===\n")
    
    checks = [
        ("PyTorch", check_torch),
        ("Transformers", check_transformers),
        ("OpenCV", check_opencv),
        ("NumPy", check_numpy),
        ("Pandas", check_pandas),
        ("ONNX", check_onnx),
        ("ONNX Runtime", check_onnxruntime),
        ("gRPC", check_grpc),
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        print(f"\n[{name}]")
        if check_func():
            passed += 1
    
    print(f"\n=== 验证结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有已安装的依赖都正常工作!")
    else:
        print(f"⚠ {total - passed} 个依赖存在问题")

if __name__ == "__main__":
    main()