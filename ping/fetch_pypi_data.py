import requests
from bs4 import BeautifulSoup, Tag
import pandas as pd
from urllib.parse import urljoin
import time

def fetch_pypi_packages_to_excel(url, output_file='pypi_packages.xlsx'):
    """
    获取PyPI简单索引页面上的所有包名，并将其保存到Excel文件中
    
    Args:
        url (str): PyPI简单索引页面的URL
        output_file (str): 输出的Excel文件名
    """
    print(f"正在访问URL: {url}")
    
    # 发送HTTP请求获取页面内容
    response = requests.get(url)
    response.raise_for_status()  # 如果请求失败会抛出异常
    
    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 提取所有的包链接
    links = soup.find_all('a')
    
    # 存储包信息的列表
    packages = []
    
    print(f"找到 {len(links)} 个包")
    
    # 遍历所有链接提取包名
    for i, link in enumerate(links):
        # 确保link是Tag对象
        if isinstance(link, Tag):
            package_name = link.get_text().strip()
            href = link.get('href')
            if href:
                # 确保href是字符串类型
                href_str = str(href) if href else ''
                package_url = urljoin(url, href_str)
            else:
                package_url = ''
            
            # 添加到列表中
            packages.append({
                'package_name': package_name,
                'package_url': package_url
            })
        
        # 每处理100个包显示一次进度
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1} 个包...")
    
    # 创建DataFrame
    df = pd.DataFrame(packages)
    
    # 保存到Excel文件
    df.to_excel(output_file, index=False)
    
    print(f"数据已保存到 {output_file}")
    print(f"总共处理了 {len(packages)} 个包")
    
    return df

if __name__ == "__main__":
    # 目标URL
    url = "https://art-pub.eng.t-head.cn/artifactory/api/pypi/ptgai-pypi_ppu_ubuntu_cu126_index/simple/"
    
    # 执行函数并将结果保存到Excel
    try:
        df = fetch_pypi_packages_to_excel(url)
        print("执行成功！")
        print(df.head())  # 显示前几行数据
    except Exception as e:
        print(f"执行过程中出现错误: {e}")