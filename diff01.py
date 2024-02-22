import torch
import streamlit as st
from PIL import Image
from diffusers import DiffusionPipeline as DP

def text_to_image(text,diffuser_model):
    dp = DP.from_pretrained("runwaym1/stable-diffusion-v1-5",)
    image_data = dp(text).images[0]
    image = Image.fromarray(image_data)
    image.show()

if __name__ =="__main__":
    input_text = "Hello, World!"
    diffusers_model = "examole_diffusers_model!"
    text_to_image(input_text, diffusers_model)
