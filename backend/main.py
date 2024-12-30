from fastapi import FastAPI
import uvicorn
from api import linear_regression_router

app = FastAPI()

app.include_router(linear_regression_router)

@app.get("/")
def root():
    return {"message": "Welcome to ML Class!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8800, reload=True)