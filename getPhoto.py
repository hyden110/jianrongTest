from fastapi import FastAPI
from fastapi.responses import FileResponse
import os

app = FastAPI(title="图片上传API", version="1.0.0")

# 定义图片路径
IMAGE_PATH = "src/main/resources/zhaji.jpg"

@app.get("/api/v1/photo/upload", tags=["Photo"])
async def upload_photo():
    """
    接收API请求，上传本地的zhaji.jpg图片
    
    Returns:
        FileResponse: 返回本地的zhaji.jpg图片文件
    """
    # 检查图片文件是否存在
    if not os.path.exists(IMAGE_PATH):
        return {"error": "图片文件不存在", "file_path": IMAGE_PATH}
    
    # 返回图片文件
    return FileResponse(
        IMAGE_PATH,
        media_type="image/jpeg",
        filename="zhaji.jpg"
    )

if __name__ == "__main__":
    import uvicorn
    # 在阿里云ECS上运行，绑定到所有网络接口
    uvicorn.run(app, host="0.0.0.0", port=8002)