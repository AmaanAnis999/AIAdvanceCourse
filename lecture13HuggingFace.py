# THESE ARE DOWNLOADING THINGS WORTH GBs ITS BETTER TO RUN THESE IN COLLAB NOTEBOOK

# import openai
# openai.api_key='sk-k5r-L6EIf6wnULYd8ThiYLGKJ4fdd0up5rkCot7wycT3BlbkFJmgzuxkmtXcsGN44I_RX7k4I59GueoV7x9LG9Cnfr4A'

# import io
# import requests
# import PIL
# from PIL import Image

# def generate_image_with_text_prompt(text_prompt):
#     """
#     Generate an image using OpenAI's DALL-E model based on the provided text prompt.

#     :param text_prompt: The prompt to generate the image.
#     :return: PIL Image object.
#     """
#     number_of_images=1
#     # using dall-e model

#     response=openai.images.generate(
#         prompt=text_prompt,
#         n=number_of_images,
#         size="256x256"
#     )

#     # get image url from response

#     image_url=response.data[0].url

#     # download the image content and convert it into pil image
#     image_content=requests.get(image_url).content
#     image=Image.open(io.BytesIO(image_content))

#     return image

# prompt = input("Enter your prompt to generate the image: ")
# image=generate_image_with_text_prompt(prompt)


# def generate_images_with_text_prompt(text_prompt):
#     """
#     Generate an image using OpenAI's DALL-E model based on the provided text prompt.

#     :param text_prompt: The prompt to generate the image.
#     :return: PIL Image object.
#     """
#     number_of_images=5
#     # using dall-e model
#     responses=openai.images.generate(
#         prompt=text_prompt,
#         n=number_of_images,
#         size="256x256"
#     )
    
#     # get image url
#     images=[]
#     for response in response.data:
#         image_url=response.url
#         image_content=requests.get(image_url).content
#         image=Image.open(io.BytesIO(image_content))
#         images.append(image)
#     return images

# prompt = input("Enter your prompt to generate the image: ")
# images=generate_images_with_text_prompt(prompt)

# # from IPython.display import display
# # for image in images:
# #     display(image)

# def generate_image_with_text_prompt_dall_e_3(text_prompt):
#     """
#     Generate an image using OpenAI's DALL-E model based on the provided text prompt.

#     :param text_prompt: The prompt to generate the image.
#     :return: PIL Image object.
#     """
#     number_of_images=1

#     response=openai.images.generate(
#         model="dall-e-3",
#         prompt=text_prompt,
#         n=number_of_images,
#         size="1024x1024"
#     )

#     image_url=response.data[0].url

#     image_content=requests.get(image_url).content
#     image=Image.open(io.BytesIO(image_content))

#     return image

# prompt=input('enter your prompt to generate the image: ')
# image=generate_image_with_text_prompt_dall_e_3(prompt)


# HUGGING FACE

# pip install transformers

import transformers
from transformers import pipeline

# pipe=pipeline("text-classification")
# pipe('this movie is very boring')

# pipe=pipeline(model="roberta-large-mnli")
# pipe('this movie is very boring')

# pipe = pipeline("sentiment-analysis")
# pipe(["This restaurant is awesome", "This restaurant is awful"])
# pipe("I don't know where I am going")

# text to be summarized
# input_text = "Start by providing your text input. It could be a sentence or a paragraph.\n Tokenization: The input is tokenized, which means breaking it down into smaller units like words or subwords.\n Tokens are the building blocks for NLP models\n.Model: The tokenized input is passed through a pre-trained NLP model. \n Hugging Face offers a wide range of models for different NLP tasks, such as sentiment analysis,\n  question answering, and text generation.Prediction/Output: The model processes the tokenized input and generates \n a prediction or output specific to the task. For example, if it's sentiment analysis, \n it could predict whether the input is positive or negative"
# print(input_text)

# # use bart in pytorch
# summarizer=pipeline("summarization")
# summarizer("Text Input: Start by providing your text input. It could be a sentence or a paragraph.Tokenization: The input is tokenized, which means breaking it down into smaller units like words or subwords. Tokens are the building blocks for NLP models.Model: The tokenized input is passed through a pre-trained NLP model. Hugging Face offers a wide range of models for different NLP tasks, such as sentiment analysis, question answering, and text generation.Prediction/Output: The model processes the tokenized input and generates a prediction or output specific to the task. For example, if it's sentiment analysis, it could predict whether the input is positive or negative.", min_length=5, max_length=30)

# NER=named entity recognition, its used in natural language processing(NLP)
# this program will tell me what is every word in the sentence telling us like "Amaan" is name etc.
# nlp=pipeline("ner")
# example="My name is Amaan"

# ner_results=nlp(example)
# print(ner_results)

# # IMAGE CLASSIFICATION
# # upload an image and change the path
# Displaying the image
# from PIL import Image

# # Specify the path to your PNG image
# image_path = '/content/dog.png'

# # Open the image using PIL
# image = Image.open(image_path)

# # Display the image
# image

# from transformers import pipeline

# classifier=pipeline(model="microsoft/beit-base-patch16-224-pt22k-ft22k")
# classifier("/content/dog.png")

## pip install diffusers (run this command for this program)

# import transformers
# from transformers import pipeline
# from diffusers import StableDiffusionPipeline
# import torch

# model_id="runwayml/stable-diffusion-v1-5"
# pipe=StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16)
# pipe=pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on Mars"

# image=pipe(prompt).images[0]
# # save the generated image
# image.save("astronaut_rides_horse.png")

# completed