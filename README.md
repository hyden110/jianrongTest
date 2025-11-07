# Vehicle Service

这是一个用Java实现的微服务，包含车牌号查询和图片上传功能，并注册到阿里云MSE Nacos。

## 功能说明

1. 车牌号查询服务 (`/api/v1/plate/check`)
   - 支持POST和GET方法
   - 返回固定结果{"result": "OK", "plate_number": "车牌号"}

2. 图片上传服务 (`/api/v1/photo/upload`)
   - 返回本地的zhaji.jpg图片

## 部署说明

### 1. 环境要求
- JDK 11或更高版本
- Maven 3.6或更高版本

### 2. 编译打包
```bash
mvn clean package
```

### 3. 运行服务
```bash
java -jar target/vehicle-service-1.0.0.jar
```

### 4. Nacos配置
服务会自动注册到以下Nacos服务器：
- 地址: mse-c65df2115-nacos-ans.mse.aliyuncs.com
- 端口: 8848

## API使用

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

## 配置说明

在`src/main/resources/application.yml`中配置了以下参数：
- 服务端口: 8002
- Nacos服务器地址和端口
- 应用名称: vehicle-service