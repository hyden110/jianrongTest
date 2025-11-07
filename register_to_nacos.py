import nacos
import socket
import time
import threading

# Nacos配置
NACOS_SERVER_ADDR = "mse-c65df2115-p.nacos-ans.mse.aliyuncs.com:8848"
SERVICE_NAMESPACE = "public"  # 命名空间为public
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

def register_services():
    """注册服务到Nacos"""
    try:
        # 创建Nacos客户端
        if ACCESS_KEY and SECRET_KEY:
            # 带认证的客户端
            client = nacos.NacosClient(NACOS_SERVER_ADDR, namespace=SERVICE_NAMESPACE, 
                                     username=ACCESS_KEY, password=SECRET_KEY)
        else:
            # 不带认证的客户端
            client = nacos.NacosClient(NACOS_SERVER_ADDR, namespace=SERVICE_NAMESPACE)
        
        # 获取本机IP
        ip = get_local_ip()
        print(f"本机IP: {ip}")
        
        # 注册车牌号查询服务 (queryData.py运行在8001端口)
        client.add_naming_instance(
            service_name="plate-service",
            ip=ip,
            port=8001,
            group_name=SERVICE_GROUP
        )
        print(f"车牌号查询服务注册成功: {ip}:8001")
        
        # 注册图片上传服务 (getPhoto.py运行在8002端口)
        client.add_naming_instance(
            service_name="photo-service",
            ip=ip,
            port=8002,
            group_name=SERVICE_GROUP
        )
        print(f"图片上传服务注册成功: {ip}:8002")
        
        print("所有服务注册成功!")
        return client, ip
        
    except Exception as e:
        print(f"注册服务时出错: {e}")
        return None, None

def send_heartbeats(client, ip):
    """发送心跳包保持服务在线"""
    if not client:
        return
        
    try:
        while True:
            # 发送车牌号查询服务心跳
            client.send_heartbeat(
                service_name="plate-service",
                ip=ip,
                port=8001,
                group_name=SERVICE_GROUP
            )
            
            # 发送图片上传服务心跳
            client.send_heartbeat(
                service_name="photo-service",
                ip=ip,
                port=8002,
                group_name=SERVICE_GROUP
            )
            
            print(f"心跳发送成功: {ip}")
            time.sleep(5)  # 每5秒发送一次心跳
            
    except Exception as e:
        print(f"发送心跳时出错: {e}")

def main():
    """主函数"""
    print("开始注册服务到Nacos...")
    
    # 注册服务
    client, ip = register_services()
    
    if client and ip:
        # 启动心跳线程
        heartbeat_thread = threading.Thread(target=send_heartbeats, args=(client, ip))
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
        
        # 保持主程序运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("程序已停止")
    else:
        print("服务注册失败")

if __name__ == "__main__":
    main()