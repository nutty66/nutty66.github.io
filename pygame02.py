import torch
from diffusers import DiffusionPipeline as DP
from PIL import Image, ImageDraw, ImageFont

def text_to_image(text, diffusers_model):
    #diffusers = diffusers.load_diffusers(diffusers_model)
    #image_data = diffusers.generate((text))
    #image = Image.fromarray(image_data)
    #image.show()

    dp = DP.from_pretrained("runwaym1/stable-diffusion-v1-5")
    image_data= dp(text.image[0])
    image = Image.fromarray(image_data)
    image.show()
    if __name__ =="__main__":

        input_text = "Hello, World!"
        diffusers_model = "examole_diffusers_model!"
        text_to_image(input_text, diffusers_model)
