#!/usr/bin/env python3
"""
已安装依赖检查脚本
验证通过PyPI镜像源安装的依赖是否能够正常使用
"""

def check_package(package_name, import_name=None):
    """检查包是否已安装并能正常导入"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
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
    print("=== 已安装依赖检查 ===\n")
    
    # 需要检查的包列表
    packages = [
        ("numpy", None),
        ("scipy", None),
        ("pandas", None),
        ("scikit-learn", "sklearn"),
        ("onnx", None),
        ("onnxruntime", None),
        ("opencv-python", "cv2"),
        ("opencv-contrib-python", "cv2"),
        ("tokenizers", None),
        ("sentencepiece", None),
        ("grpcio", None),
        ("torch", None),
        ("torchvision", None),
        ("transformers", None),
        ("xformers", None),
        ("flash_attn", None),
        ("vllm", None),
    ]
    
    installed_count = 0
    total_count = len(packages)
    
    for package_name, import_name in packages:
        if check_package(package_name, import_name):
            installed_count += 1
    
    print(f"\n=== 检查结果 ===")
    print(f"已安装: {installed_count}/{total_count}")
    
    if installed_count == total_count:
        print("🎉 所有依赖都已安装!")
    else:
        print(f"⚠ {total_count - installed_count} 个依赖未安装或导入失败")

if __name__ == "__main__":
    main()