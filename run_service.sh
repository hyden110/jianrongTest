#!/bin/bash

# 检查是否安装了Maven
if ! command -v mvn &> /dev/null
then
    echo "Maven未安装，正在安装Maven..."
    
    # 检查操作系统类型
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS系统
        if command -v brew &> /dev/null
        then
            # 临时取消清华源设置
            unset HOMEBREW_BREW_GIT_REMOTE
            unset HOMEBREW_CORE_GIT_REMOTE
            brew install maven
        else
            echo "请先安装Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
    else
        # 其他系统，尝试使用apt-get (Ubuntu/Debian)
        if command -v apt-get &> /dev/null
        then
            sudo apt-get update
            sudo apt-get install -y maven
        else
            echo "无法自动安装Maven，请手动安装Maven 3.6+"
            exit 1
        fi
    fi
else
    echo "Maven已安装"
fi

# 编译项目
echo "正在编译项目..."
mvn clean package

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo "编译成功，正在启动服务..."
    
    # 查找生成的jar文件
    JAR_FILE=$(find target -name "*.jar" -type f | head -n 1)
    
    if [ -n "$JAR_FILE" ]; then
        echo "启动服务: java -jar $JAR_FILE"
        java -jar "$JAR_FILE"
    else
        echo "未找到生成的jar文件"
        exit 1
    fi
else
    echo "编译失败"
    exit 1
fi