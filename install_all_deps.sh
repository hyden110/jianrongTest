#!/bin/bash

# 全依赖安装脚本
# 检查并安装所有指定的依赖包

# 设置PyPI镜像源
PIP_INDEX_URL="https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/ubuntu_cu128/simple/"

# 需要安装的依赖包列表（按依赖顺序排列）
PACKAGES=(
    "numpy"
    "scipy"
    "pandas"
    "polars"
    "scikit-learn"
    "scikit-image"
    "onnx"
    "onnxruntime"
    "opencv-python"
    "opencv-contrib-python"
    "opencv-python-headless"
    "dlib"
    "tokenizers"
    "sentencepiece"
    "grpcio"
    "gradio"
    "streamlit"
    "torch"
    "torchvision"
    "torchaudio"
    "pytorch-lightning"
    "fastai"
    "transformers"
    "xformers"
    "flash_attn"
    "triton"
    "cupy-cuda12x"
    "vllm"
    "tensorflow"
    "keras"
    "paddlepaddle"
    "paddlepaddle-gpu"
    "paddledet"
    "paddlex"
    "paddleocr"
    "openvino"
    "openvino-dev"
    "tensorrt"
    "tensorrt_cu13*"
    "nvidia-cuda-runtime-cu12"
    "nvidia-cublas-cu12"
    "nvidia-cudnn-cu12"
    "nvidia-nccl-cu12"
    "nvidia-cuda-nvrtc-cu12"
    "nvidia-cuda-cupti-cu12"
    "nvidia-nvjitlink-cu12"
    "nvidia-cuda-runtime-cu13"
    "nvidia-cublas-cu11"
    "nvidia-cudnn-cu11"
    "nvidia-cuda-nvrtc-cu11"
    "ray"
)

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查包是否已安装
check_package() {
    local package=$1
    python -c "import $package" 2>/dev/null
    return $?
}

# 安装单个包
install_package() {
    local package=$1
    echo -e "${YELLOW}正在安装 $package...${NC}"
    
    # 特殊处理某些包的安装
    case $package in
        "opencv-python"|"opencv-contrib-python"|"opencv-python-headless")
            pip install "$package" -i "$PIP_INDEX_URL" --timeout 300 --no-cache-dir
            ;;
        "scikit-learn")
            pip install scikit-learn -i "$PIP_INDEX_URL" --timeout 300 --no-cache-dir
            ;;
        "scikit-image")
            pip install scikit-image -i "$PIP_INDEX_URL" --timeout 300 --no-cache-dir
            ;;
        "pytorch-lightning")
            pip install pytorch-lightning -i "$PIP_INDEX_URL" --timeout 300 --no-cache-dir
            ;;
        "nvidia-"*)
            # NVIDIA相关包可能需要特殊处理
            pip install "$package" -i "$PIP_INDEX_URL" --timeout 300 --no-cache-dir --extra-index-url https://pypi.org/simple/
            ;;
        "tensorrt_cu13*")
            # TensorRT特殊处理
            pip install tensorrt -i "$PIP_INDEX_URL" --timeout 300 --no-cache-dir
            ;;
        *)
            pip install "$package" -i "$PIP_INDEX_URL" --timeout 300 --no-cache-dir
            ;;
    esac
    
    local result=$?
    if [ $result -eq 0 ]; then
        echo -e "${GREEN}✓ $package 安装成功${NC}"
        return 0
    else
        echo -e "${RED}✗ $package 安装失败${NC}"
        return 1
    fi
}

# 验证包功能
verify_package() {
    local package=$1
    local import_name=$2
    
    # 如果没有指定导入名称，使用包名
    if [ -z "$import_name" ]; then
        import_name=$package
    fi
    
    # 特殊处理某些包的导入名称
    case $package in
        "nvidia-cuda-runtime-cu12")
            import_name="nvidia.cuda_runtime"
            ;;
        "nvidia-cublas-cu12")
            import_name="nvidia.cublas"
            ;;
        "nvidia-cudnn-cu12")
            import_name="nvidia.cudnn"
            ;;
        "nvidia-nccl-cu12")
            import_name="nvidia.nccl"
            ;;
        "nvidia-cuda-nvrtc-cu12")
            import_name="nvidia.nvrtc"
            ;;
        "nvidia-cuda-cupti-cu12")
            import_name="nvidia.cupti"
            ;;
        "nvidia-nvjitlink-cu12")
            import_name="nvidia.nvjitlink"
            ;;
        "pytorch-lightning")
            import_name="pytorch_lightning"
            ;;
        "scikit-image")
            import_name="skimage"
            ;;
        "paddlepaddle"|"paddlepaddle-gpu")
            import_name="paddle"
            ;;
        "tensorrt_cu13*")
            import_name="tensorrt"
            ;;
    esac
    
    python -c "import $import_name; print('$import_name version:', $import_name.__version__ if hasattr($import_name, '__version__') else 'unknown')" 2>/dev/null
    return $?
}

# 获取包的导入名称
get_import_name() {
    local package=$1
    case $package in
        "opencv-python"|"opencv-contrib-python"|"opencv-python-headless")
            echo "cv2"
            ;;
        "scikit-learn")
            echo "sklearn"
            ;;
        "scikit-image")
            echo "skimage"
            ;;
        "pytorch-lightning")
            echo "pytorch_lightning"
            ;;
        "nvidia-cuda-runtime-cu12")
            echo "nvidia.cuda_runtime"
            ;;
        "nvidia-cublas-cu12")
            echo "nvidia.cublas"
            ;;
        "nvidia-cudnn-cu12")
            echo "nvidia.cudnn"
            ;;
        "nvidia-nccl-cu12")
            echo "nvidia.nccl"
            ;;
        "nvidia-cuda-nvrtc-cu12")
            echo "nvidia.nvrtc"
            ;;
        "nvidia-cuda-cupti-cu12")
            echo "nvidia.cupti"
            ;;
        "nvidia-nvjitlink-cu12")
            echo "nvidia.nvjitlink"
            ;;
        "nvidia-cuda-runtime-cu13")
            echo "nvidia.cuda_runtime"
            ;;
        "nvidia-cublas-cu11")
            echo "nvidia.cublas"
            ;;
        "nvidia-cudnn-cu11")
            echo "nvidia.cudnn"
            ;;
        "nvidia-cuda-nvrtc-cu11")
            echo "nvidia.nvrtc"
            ;;
        "paddlepaddle"|"paddlepaddle-gpu")
            echo "paddle"
            ;;
        "tensorrt_cu13*")
            echo "tensorrt"
            ;;
        *)
            echo "$package"
            ;;
    esac
}

# 重新加载Python路径
reload_python_path() {
    echo -e "${BLUE}重新加载Python路径...${NC}"
    python -c "import importlib; import sys; importlib.invalidate_caches()"
}

# 主函数
main() {
    echo -e "${BLUE}=== 全依赖检查和安装脚本 ===${NC}"
    echo "PyPI镜像源: $PIP_INDEX_URL"
    echo
    
    local installed_count=0
    local total_count=${#PACKAGES[@]}
    local failed_packages=()
    
    # 重新加载Python路径
    reload_python_path
    
    # 检查每个包
    for package in "${PACKAGES[@]}"; do
        echo -e "${BLUE}检查 $package...${NC}"
        
        # 获取导入名称
        import_name=$(get_import_name "$package")
        
        if check_package "$import_name"; then
            echo -e "${GREEN}✓ 已安装${NC}"
            
            # 验证功能
            if verify_package "$package" "$import_name"; then
                echo -e "  ${GREEN}✓ 功能验证通过${NC}"
            else
                echo -e "  ${YELLOW}⚠ 功能验证失败${NC}"
            fi
            
            ((installed_count++))
        else
            echo -e "${RED}✗ 未安装${NC}"
            
            # 尝试安装
            if install_package "$package"; then
                # 安装成功后再次验证
                reload_python_path
                if check_package "$import_name"; then
                    echo -e "  ${GREEN}✓ 安装验证通过${NC}"
                    ((installed_count++))
                else
                    echo -e "  ${RED}✗ 安装后仍无法导入${NC}"
                    failed_packages+=("$package")
                fi
            else
                failed_packages+=("$package")
            fi
        fi
        echo
    done
    
    # 输出总结
    echo -e "${BLUE}=== 安装总结 ===${NC}"
    echo "总计: $total_count 个包"
    echo "已安装: $installed_count 个包"
    
    if [ ${#failed_packages[@]} -gt 0 ]; then
        echo -e "${RED}安装失败的包:${NC}"
        for pkg in "${failed_packages[@]}"; do
            echo "  - $pkg"
        done
        echo
        echo -e "${YELLOW}提示: 某些包可能需要特殊安装步骤或编译环境${NC}"
        echo -e "${YELLOW}提示: 请确保已安装必要的系统依赖和编译工具${NC}"
        echo -e "${YELLOW}提示: NVIDIA相关包可能需要CUDA环境支持${NC}"
        echo -e "${YELLOW}提示: AI框架相关包可能需要大量内存和时间安装${NC}"
    else
        echo -e "${GREEN}🎉 所有包都已成功安装!${NC}"
    fi
    
    return ${#failed_packages[@]}
}

# 运行主函数
main