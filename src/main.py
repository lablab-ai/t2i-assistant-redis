# uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
from fastapi import FastAPI, HTTPException, UploadFile, status
from PIL import Image
from pydantic import BaseModel

from src.model import CLIP

# User can look for similar images or prompts
SEARCH_MODES = ["image", "prompt"]


class SearchBody(BaseModel):
    description: str


app = FastAPI()

clip = CLIP()


@app.post("/search/image/{mode}/{n}")
async def search_image(mode: str, n: int, image: UploadFile):
    if mode not in SEARCH_MODES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Mode must be one of {' '.join(SEARCH_MODES)}",
        )

    if type(n) != int:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="n must be an integer",
        )

    # check if image is valid
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File is not an image",
        )

    image = Image.open(image.file)

    # embed image using CLIP
    clip.encode_image(image)

    # search for similar images/prompts

    # based on mode, return either image or prompt

    return {"mode": mode}


@app.post("/search/description/{mode}/{n}")
async def search_description(mode: str, n: int, body: SearchBody):
    if mode not in SEARCH_MODES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Mode must be one of {' '.join(SEARCH_MODES)}",
        )

    if type(n) != int:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="n must be an integer",
        )

    # embed description using CLIP
    clip.encode_text(body.description)

    # search for similar images/prompts

    # based on mode, return either image or prompt

    return {"mode": mode}
