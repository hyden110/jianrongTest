# Nacos服务注册说明

## 配置信息

- Nacos服务器地址: mse-c65df2115-p.nacos-ans.mse.aliyuncs.com
- 端口: 8848
- 命名空间: public
- 服务组: DEFAULT_GROUP

## 问题说明

在尝试将Python服务注册到Nacos时，遇到了"Insufficient privilege"（权限不足）错误。这说明Nacos服务器启用了认证机制，需要提供有效的访问凭证。

## 解决方案

### 1. 获取Nacos访问密钥

要成功注册服务到Nacos，您需要获取访问密钥：
1. 登录阿里云控制台
2. 进入MSE（Microservice Engine）控制台
3. 找到对应的Nacos实例
4. 在实例详情页面获取或创建访问密钥（Access Key和Secret Key）

### 2. 配置认证信息

获取到访问密钥后，需要在[register_to_nacos.py](file:///Users/linzhou/project/rag/vlrag/register_to_nacos.py)文件中配置：

```python
# Nacos配置
NACOS_SERVER_ADDR = "mse-c65df2115-p.nacos-ans.mse.aliyuncs.com:8848"
SERVICE_NAMESPACE = "public"
SERVICE_GROUP = "DEFAULT_GROUP"

# 配置访问密钥
ACCESS_KEY = "您的Access Key"
SECRET_KEY = "您的Secret Key"
```

### 3. 运行服务注册脚本

确保服务已启动：
- queryData.py运行在8001端口
- getPhoto.py运行在8002端口

然后运行注册脚本：
```bash
python register_to_nacos.py
```

### 4. 验证注册结果

注册成功后，可以在Nacos控制台的"服务管理" -> "服务列表"中查看到以下服务：
- plate-service (端口8001)
- photo-service (端口8002)

## 手动注册方式

如果没有编程方式注册的条件，也可以通过Nacos控制台手动注册服务：

1. 登录阿里云MSE控制台
2. 进入Nacos实例管理页面
3. 点击"服务管理" -> "服务列表"
4. 点击"创建服务"
5. 填写服务信息：
   - 服务名：plate-service
   - 分组：DEFAULT_GROUP
   - 命名空间：public
   - 保护阈值：0
6. 添加实例：
   - IP：您的服务器IP
   - 端口：8001
   - 健康状态：健康
7. 重复步骤4-6注册photo-service服务（端口8002）

## 注意事项

1. 确保您的服务器IP可以从Nacos服务器访问
2. 确保8001和8002端口在安全组中已开放
3. 注册脚本会持续发送心跳包保持服务在线，不要停止脚本运行
4. 如果使用了RAM角色，请确保角色具有Nacos相关权限