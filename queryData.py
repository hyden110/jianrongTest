from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="车牌号查询API", version="1.0.0")

class PlateNumberRequest(BaseModel):
    plate_number: str

class PlateNumberResponse(BaseModel):
    result: str
    plate_number: str

@app.post("/api/v1/plate/check", response_model=PlateNumberResponse, tags=["Plate"])
async def check_plate_number(req: PlateNumberRequest):
    """
    处理车牌号请求，返回固定结果"OK"
    
    Args:
        req (PlateNumberRequest): 包含车牌号的请求体
        
    Returns:
        PlateNumberResponse: 固定返回结果{"result": "OK", "plate_number": "车牌号"}
    """
    # 这里可以添加实际的车牌号处理逻辑
    # 目前按照需求返回固定结果"OK"
    return PlateNumberResponse(result="OK", plate_number=req.plate_number)

@app.get("/api/v1/plate/check", response_model=PlateNumberResponse, tags=["Plate"])
async def check_plate_number_get(plate_number: str = Query(..., description="车牌号")):
    """
    通过GET方法处理车牌号请求，返回固定结果"OK"
    
    Args:
        plate_number (str): 车牌号参数
        
    Returns:
        PlateNumberResponse: 固定返回结果{"result": "OK", "plate_number": "车牌号"}
    """
    return PlateNumberResponse(result="OK", plate_number=plate_number)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)