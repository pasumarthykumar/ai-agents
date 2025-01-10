from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.agents.web_agent import web_agent
from app.agents.text_agent import TextAgent
from app.agents.rag_agent import FileAgent
from app.agents.image_agent import ImageAgent
from app.agents.video_agent import VideoAgent
from fastapi.responses import StreamingResponse
from gtts import gTTS
from io import BytesIO

router = APIRouter()

# Define the input schema
class WebQueryRequest(BaseModel):
    query: str

# Define the input schema
class TextQuery(BaseModel):
    query: str

class FileQuery(BaseModel):
    query: str

class FileUploadQuery(BaseModel):
    query: str
    file: UploadFile

# Pydantic model for the image generation request
class ImageQuery(BaseModel):
    prompt: str


@router.post("/web-query/")
async def web_query(request: WebQueryRequest):
    """
    Endpoint to handle web-based queries using the Web Agent.
    """
    try:
        response = web_agent.run(f"Search the web for: {request.query}")
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error handling web query: {str(e)}")
    
# Define the route for handling text-based queries
@router.post("/text-query/")
async def handle_text_query(text_query: TextQuery):
    """
    Handles general text queries by using the TextAgent.
    """
    try:
        agent = TextAgent()
        response = agent.process_text_query(text_query.query)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.post("/file-query/")
async def handle_file_query(query: str = Form(...), file: UploadFile = File(...)):
    """
    Handles queries based on uploaded files using RAG.
    """
    try:
        agent = FileAgent()
        response = agent.process_file_query(query, file)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    

@router.post("/image-query/")
async def handle_image_query(query: ImageQuery):
    """
    Endpoint to generate an image based on a user's prompt.
    """
    try:
        agent = ImageAgent()
        return agent.generate_image(query.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
@router.post("/video-query/")
async def handle_video_query(video: UploadFile):
    """
    Handles queries based on uploaded video files.
    """
    try:
        # Initialize VideoAgent
        agent = VideoAgent()
        
        # Process video query
        response = agent.process_video_query(video)
        return response.content
       
        # Convert response to audio
        tts = gTTS(text=response, lang="en")
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        # Return the audio as a response
        return StreamingResponse(audio_buffer, media_type="audio/mpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video query: {str(e)}")


