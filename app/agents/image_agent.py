from openai import OpenAI
import requests
from io import BytesIO
from fastapi.responses import StreamingResponse
from fastapi import HTTPException

client = OpenAI()

class ImageAgent:
    """
    Handles image generation using OpenAI's DALL·E.
    """

    @staticmethod
    def generate_image(prompt: str):
        """
        Generate an image based on the prompt and return it as a binary response.
        """
        try:
            # Use OpenAI's updated DALL·E API for image generation
          
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            print(response)
            # Get the image URL from the response
            image_url = response.data[0].url
            print(image_url)
            # Fetch the image content from the URL
            image_response = requests.get(image_url)
            if image_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to fetch the generated image")

            # Return the image as a streaming response
            return StreamingResponse(
                BytesIO(image_response.content),
                media_type="image/png"
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")
