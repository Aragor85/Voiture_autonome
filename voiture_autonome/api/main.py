from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from model_utils import load_model_and_predict

app = FastAPI()

@app.post("/predict/")
async def predict_mask(file: UploadFile = File(...)):
    image_bytes = await file.read()
    prediction = load_model_and_predict(image_bytes)
    return JSONResponse(content={"prediction": prediction.tolist()})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
