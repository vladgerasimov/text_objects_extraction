from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.dependencies.extractors import get_bert_extractor
from src.endpoints.extract_objects import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_bert_extractor()
    yield


app = FastAPI(title="model_explorer_api", lifespan=lifespan)


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, reload=True)  # pyright: ignore [reportUnknownMemberType]
