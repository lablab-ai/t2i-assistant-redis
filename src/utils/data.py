import os
from uuid import uuid4

import pandas as pd
from redis.commands.search.field import TextField, VectorField

EMBEDDING_DIM = 1024


def embed_record(clip, caption):
    caption_features = clip.encode_text(caption).squeeze()

    return caption_features.cpu().detach().numpy()


def index_data(redis_client, clip):
    redis_client.ft().dropindex()

    DATA_DIR = os.path.join("data")
    df = pd.read_csv(os.path.join(DATA_DIR, "captions.csv"))

    redis_client.ft().create_index(
        [
            TextField("image"),
            TextField("caption"),
            VectorField(
                "caption_features",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": EMBEDDING_DIM,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        ]
    )

    selected_data = (
        df.iloc[::5, :]
        .iloc[:7]
        .apply(
            lambda x: (x["image"], x["caption"], embed_record(clip, x["caption"])),
            axis=1,
        )
        .to_numpy()
    )

    pipe = redis_client.pipeline()
    i = 0
    for img_filename, caption, caption_features in selected_data:
        pipe.hset(
            uuid4().hex,
            mapping={
                "image": img_filename,
                "caption": caption,
                "caption_features": caption_features.tobytes(),
            },
        )
        i += 1
    pipe.execute()
