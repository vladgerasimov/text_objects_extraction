import uvicorn
from fastapi import FastAPI

from src.endpoints.extract_objects import router

app = FastAPI(title="model_explorer_api")


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, reload=True)  # pyright: ignore [reportUnknownMemberType]
