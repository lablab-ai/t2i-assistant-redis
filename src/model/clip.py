from typing import List

import open_clip
import torch
from PIL import Image


# https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
class CLIP:
    def __init__(
        self, model_name="hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K", device="cpu"
    ):
        model, _, preprocess_img = open_clip.create_model_and_transforms(model_name)

        self.device = device

        self.model = model.to(self.device)
        self.preprocess_img = preprocess_img
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_image(self, image: Image.Image | List[Image.Image], normalize=True):
        processed_img = (
            torch.stack([self.preprocess_img(img).to(self.device) for img in image])
            if type(image) == list
            else self.preprocess_img(image).to(self.device)
        )

        if processed_img.dim() == 3:
            processed_img = processed_img.unsqueeze(0)

        image_features = self.model.encode_image(processed_img)

        if normalize:
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features

    def encode_text(self, text: str | List[str], normalize=True):
        text = self.tokenizer(text).to(self.device)

        text_features = self.model.encode_text(text)

        if normalize:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features
