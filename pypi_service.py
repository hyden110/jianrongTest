"""
PyPI数据获取服务
提供从PyPI索引链接获取数据并保存到Excel文件的功能
"""
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from urllib.parse import urljoin, urlparse
import re


def fetch_pypi_data(url: str) -> List[Dict[str, Any]]:
    """
    从PyPI索引链接获取数据
    
    Args:
        url: PyPI索引链接
        
    Returns:
        包含包信息的字典列表
    """
    try:
        # 发送GET请求获取页面内容
        response = requests.get(url)
        response.raise_for_status()
        
        # 解析HTML内容，提取包信息
        html_content = response.text
        
        # 使用正则表达式提取包名和链接
        # 匹配 <a href="...">package_name</a> 格式
        pattern = r'<a(?:\s+[^>]*?)?\s+href=["\']([^"\']*)["\'][^>]*?>([^<]+)</a>'
        matches = re.findall(pattern, html_content)
        
        packages = []
        for link, package_name in matches:
            # 解析链接获取包的详细信息
            package_info = {
                "name": package_name.strip(),
                "link": link,
                "full_link": urljoin(url, link) if not link.startswith('http') else link
            }
            packages.append(package_info)
        
        return packages
    except Exception as e:
        raise Exception(f"获取PyPI数据失败: {str(e)}")


def save_to_excel(data: List[Dict[str, Any]], filename: str = "pypi_packages.xlsx") -> str:
    """
    将数据保存到Excel文件
    
    Args:
        data: 要保存的数据列表
        filename: 保存的文件名
        
    Returns:
        保存的文件路径
    """
    try:
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 保存到Excel文件
        output_path = Path(filename)
        df.to_excel(output_path, index=False, engine='openpyxl')
        
        return str(output_path)
    except Exception as e:
        raise Exception(f"保存Excel文件失败: {str(e)}")


def fetch_and_save_pypi_data(url: str, filename: str = "pypi_packages.xlsx") -> str:
    """
    从PyPI链接获取数据并保存到Excel文件
    
    Args:
        url: PyPI索引链接
        filename: 保存的Excel文件名
        
    Returns:
        保存的文件路径
    """
    # 获取数据
    packages = fetch_pypi_data(url)
    
    # 保存到Excel
    file_path = save_to_excel(packages, filename)
    
    return file_path