import json
import os

import requests

import streamlit as st

# Add a prompt to the app
prompt = st.text_input("Prompt")

# Add file uploader to the app
image = st.file_uploader("Upload an image")

# Add a button to the app
button = st.button("Find similar images/prompts")

# when the button is clicked
if button:
    # if the user uploaded an image
    if image:
        URL = "http://localhost:8000/search/image"
        IMG_EXT = ["jpg", "jpeg", "png"]

        file_extension = image.name.split(".")[-1]
        print(file_extension)

        if not file_extension in IMG_EXT:
            print("Invalid file extension")

        # send the image to the server (form data)
        files = {
            "image": (
                image.name,
                image.read(),
                f"image/{file_extension}",
            ),
        }

        response = requests.post(
            URL,
            files=files,
        )

        # display the response
        res = response.json()

        caption = res["caption"]
        image = os.path.join("data", "Images", res["image"])

        st.image(image, caption=caption)

    if prompt and not image:
        URL = "http://localhost:8000/search/description"
        response = requests.post(
            URL,
            data=json.dumps({"description": prompt}),
        )

        res = response.json()

        caption = res["caption"]
        image = os.path.join("data", "Images", res["image"])

        st.image(image, caption=caption)
