import requests
import pandas as pd
from urllib.parse import urlparse
import os

def download_and_save_to_excel(url, output_filename='downloaded_data.xlsx'):
    """
    从指定URL下载数据并保存到Excel文件
    
    Args:
        url (str): 要访问的URL
        output_filename (str): 输出的Excel文件名
    """
    try:
        # 发送GET请求获取数据
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功
        
        # 解析URL获取文件名
        parsed_url = urlparse(url)
        file_name = os.path.basename(parsed_url.path)
        
        # 如果没有指定文件扩展名，根据内容类型判断
        if not output_filename.endswith('.xlsx'):
            output_filename += '.xlsx'
        
        # 将响应内容保存到Excel
        # 这里假设返回的是文本数据，可以根据实际数据格式进行调整
        data = response.text
        
        # 创建DataFrame并保存到Excel
        # 如果数据是JSON格式，可以使用pd.json_normalize()
        df = pd.DataFrame([data.split('\n')])  # 简单按行分割
        
        # 如果需要更复杂的处理，可以根据实际返回的数据结构调整
        df.to_excel(output_filename, index=False, header=False)
        
        print(f"数据已成功保存到 {output_filename}")
        return output_filename
        
    except requests.exceptions.RequestException as e:
        print(f"下载数据时出错: {e}")
        return None
    except Exception as e:
        print(f"保存到Excel时出错: {e}")
        return None

# 使用示例
url = "https://art-pub.eng.t-head.cn/artifactory/api/pypi/ptgai-pypi_ppu_ubuntu_cu126_index/simple/"
output_file = download_and_save_to_excel(url, "pypi_data.xlsx")
