#!/bin/bash

# ML依赖安装脚本
# 检查并安装机器学习相关依赖包

# 设置PyPI镜像源
PIP_INDEX_URL="https://art-pub.eng.t-head.cn/artifactory/api/pypi/ptgai-pypi_ppu_ubuntu_cu128_index/simple/"

# 需要安装的依赖包列表（按依赖顺序排列）
PACKAGES=(
    "numpy"
    "scipy"
    "pandas"
    "scikit-learn"
    "onnx"
    "onnxruntime"
    "opencv-python"
    "opencv-contrib-python"
    "tokenizers"
    "sentencepiece"
    "grpcio"
    "torch"
    "torchvision"
    "transformers"
    "xformers"
    "flash_attn"
    "vllm"
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
        "opencv-python"|"opencv-contrib-python")
            pip install "$package" -i "$PIP_INDEX_URL" --timeout 300 --no-cache-dir
            ;;
        "scikit-learn")
            pip install scikit-learn -i "$PIP_INDEX_URL" --timeout 300 --no-cache-dir
            ;;
        "torch")
            # 特殊处理torch安装
            pip install torch -i "$PIP_INDEX_URL" --timeout 300 --no-cache-dir --force-reinstall
            ;;
        "torchvision")
            # 特殊处理torchvision安装
            pip install torchvision -i "$PIP_INDEX_URL" --timeout 300 --no-cache-dir --force-reinstall
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
    
    python -c "import $import_name; print('$import_name version:', $import_name.__version__ if hasattr($import_name, '__version__') else 'unknown')" 2>/dev/null
    return $?
}

# 获取包的导入名称
get_import_name() {
    local package=$1
    case $package in
        "opencv-python"|"opencv-contrib-python")
            echo "cv2"
            ;;
        "scikit-learn")
            echo "sklearn"
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
    echo -e "${BLUE}=== ML依赖检查和安装脚本 ===${NC}"
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
        echo -e "${YELLOW}提示: 对于torch相关包，可能需要先卸载再重新安装${NC}"
    else
        echo -e "${GREEN}🎉 所有包都已成功安装!${NC}"
    fi
    
    return ${#failed_packages[@]}
}

# 运行主函数
main