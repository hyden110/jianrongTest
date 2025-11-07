# Java车辆服务部署指南

## 部署步骤

### 1. 安装必要工具

#### 安装Maven (推荐方式)
```bash
# macOS (使用Homebrew)
brew install maven

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install maven

# CentOS/RHEL
sudo yum install maven
```

### 2. 编译和打包应用

```bash
# 进入项目根目录
cd /Users/linzhou/project/rag/vlrag

# 清理并编译项目
mvn clean package
```

### 3. 运行服务

```bash
# 运行打包好的JAR文件
java -jar target/vehicle-service-1.0.0.jar
```

### 4. 验证服务

服务启动后会自动注册到Nacos注册中心，可以通过以下方式验证：

1. 访问API端点：
   - 车牌号查询：`http://<服务器IP>:8002/api/v1/plate/check`
   - 图片上传：`http://<服务器IP>:8002/api/v1/photo/upload`

2. 检查Nacos注册：
   访问Nacos控制台：http://mse-c65df2115-p.nacos-ans.mse.aliyuncs.com:8848/nacos
   查看服务列表中是否包含`vehicle-service`

### 5. 后台运行

```bash
# 使用nohup后台运行
nohup java -jar target/vehicle-service-1.0.0.jar > vehicle-service.log 2>&1 &

# 或使用screen
screen -dmS vehicle-service java -jar target/vehicle-service-1.0.0.jar
```

## 配置说明

### application.yml
```yaml
server:
  port: 8002  # 服务端口

spring:
  application:
    name: vehicle-service  # 服务名称
  cloud:
    nacos:
      discovery:
        server-addr: mse-c65df2115-p.nacos-ans.mse.aliyuncs.com:8848  # Nacos地址
        namespace: public
        group: DEFAULT_GROUP
```

## API使用示例

### 车牌号查询
```bash
# POST方法
curl -X POST "http://localhost:8002/api/v1/plate/check" \
     -H "Content-Type: application/json" \
     -d '{"plateNumber": "京A12345"}'

# GET方法
curl "http://localhost:8002/api/v1/plate/check?plate_number=京A12345"
```

### 图片上传
```bash
curl "http://localhost:8002/api/v1/photo/upload" -o downloaded_image.jpg
```

## 故障排除

1. 如果服务无法启动，请检查：
   - Java版本是否为11或更高
   - Maven是否正确安装
   - 网络连接是否正常

2. 如果服务未注册到Nacos，请检查：
   - Nacos服务器地址是否正确
   - 网络防火墙是否允许访问8848端口
   - Nacos服务是否正常运行