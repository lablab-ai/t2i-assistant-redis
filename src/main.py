import numpy as np
import redis
from fastapi import FastAPI, HTTPException, UploadFile, status
from PIL import Image
from pydantic import BaseModel
from redis.commands.search.query import Query

from src.model import CLIP
from src.utils import index_data

### DATA PART ####
clip = CLIP()

import redis

redis_client = redis.Redis(
    host="redis-10292.c23738.us-east-1-mz.ec2.cloud.rlrcp.com",
    port=10292,
    password="newnative",
)

index_data(redis_client, clip)


def query_image(caption_features: np.array, n=1):
    if caption_features.dtype != np.float32:
        raise TypeError("caption_features must be of type float32")

    query = (
        Query(f"*=>[KNN {n} @caption_features $caption_features]")
        .return_fields("image", "caption")
        .dialect(2)
    )

    result = redis_client.ft().search(
        query=query, query_params={"caption_features": caption_features.tobytes()}
    )

    return result.docs


s = query_image(np.random.rand(1024).astype(np.float32))


## API PART ####
class SearchBody(BaseModel):
    description: str


app = FastAPI()


@app.post("/search/image/")
async def search_by_image(image: UploadFile):
    # check if image is valid
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File is not an image",
        )

    image = Image.open(image.file)

    # embed image using CLIP
    img_features = clip.encode_image(image)

    img_features = img_features.squeeze().cpu().detach().numpy().astype(np.float32)

    # search for similar images/prompts
    result = query_image(img_features)

    result = result[0]

    return {
        "image": result["image"],
        "caption": result["caption"],
    }


@app.post("/search/description/")
async def search_description(body: SearchBody):
    # embed description using CLIP
    caption_features = clip.encode_text(body.description)

    # cast to float32
    caption_features = (
        caption_features.squeeze().cpu().detach().numpy().astype(np.float32)
    )

    # search for similar images/prompts
    result = query_image(caption_features)

    result = result[0]

    return {
        "image": result["image"],
        "caption": result["caption"],
    }
