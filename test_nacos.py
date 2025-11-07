import requests
import socket
import time
import hashlib
import hmac
import base64

# Nacos配置
NACOS_SERVER_ADDR = "47.104.74.80:8848"
SERVICE_NAMESPACE = "public"
SERVICE_GROUP = "DEFAULT_GROUP"

# 如果Nacos启用了认证，需要配置访问密钥
# 请根据实际情况填写您的Access Key和Secret Key
ACCESS_KEY = ""
SECRET_KEY = ""

def get_local_ip():
    """获取本机IP地址"""
    try:
        # 创建一个UDP socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 连接到一个远程地址（不会真正发送数据）
        s.connect(("8.8.8.8", 80))
        # 获取本地IP地址
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"获取本机IP失败: {e}")
        return "127.0.0.1"

def sign_request(secret_key, data):
    """签名请求"""
    signature = hmac.new(secret_key.encode('utf-8'), data.encode('utf-8'), hashlib.sha1).digest()
    return base64.b64encode(signature).decode('utf-8')

def test_nacos_api():
    """测试Nacos API"""
    try:
        # 获取本机IP
        ip = get_local_ip()
        print(f"本机IP: {ip}")
        
        # 测试注册服务
        print("\n测试注册服务...")
        url = f"http://{NACOS_SERVER_ADDR}/nacos/v1/ns/instance"
        params = {
            "serviceName": "test-service",
            "ip": ip,
            "port": 8001,
            "namespaceId": SERVICE_NAMESPACE,
            "groupName": SERVICE_GROUP,
            "healthy": "true",
            "enabled": "true",
            "ephemeral": "true"
        }
        
        # 如果配置了访问密钥，则添加认证头
        headers = {}
        if ACCESS_KEY and SECRET_KEY:
            # 生成时间戳
            timestamp = str(int(time.time() * 1000))
            
            # 构造签名数据
            sign_data = f"{ACCESS_KEY}|{timestamp}"
            signature = sign_request(SECRET_KEY, sign_data)
            
            # 添加认证头
            headers["accessToken"] = ACCESS_KEY
            headers["signature"] = signature
            headers["timestamp"] = timestamp
        
        response = requests.post(url, params=params, headers=headers, timeout=10)
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        
    except Exception as e:
        print(f"测试Nacos API时出错: {e}")

if __name__ == "__main__":
    test_nacos_api()