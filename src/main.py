# uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
from fastapi import FastAPI, HTTPException, UploadFile, status
from PIL import Image
from pydantic import BaseModel


class SearchBody(BaseModel):
    description: str


app = FastAPI()


@app.post("/search/image/{mode}")
async def search_image(mode: str, image: UploadFile):
    # check if image is valid
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File is not an image",
        )

    image = Image.open(image.file)

    # embed image using CLIP

    # search for similar images/prompts

    # based on mode, return either image or prompt

    return {"mode": mode}


@app.post("/search/description/{mode}")
async def search_description(mode: str, body: SearchBody):
    # embed description using CLIP

    # search for similar images/prompts

    # based on mode, return either image or prompt

    return {"mode": mode}
