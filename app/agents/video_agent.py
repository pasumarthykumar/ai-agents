import ffmpeg
from google.cloud import speech
import io
from google.cloud import vision
import io
import cv2
import os
from fastapi import APIRouter, HTTPException, UploadFile
from langchain.chat_models import ChatOpenAI
from decouple import config


GOOGLE_APPLICATION_CREDENTIALS = config("GOOGLE_APPLICATION_CREDENTIALS") 

llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
class AudioProcessor:
    @staticmethod
    def extract_audio(video_path: str, audio_path: str):
        """
        Extract audio from the video file and save it as a separate file.
        """
        try:
            ffmpeg.input(video_path).output(audio_path).run(cmd='/opt/homebrew/bin/ffmpeg',overwrite_output=True)
        except ffmpeg.Error as e:
            raise Exception(f"Error extracting audio: {e.stderr.decode()}")


class ObjectDetection:
    @staticmethod
    def detect_objects(image_path: str):
        """
        Detect objects in an image using Google Vision API.
        """
        try:
            # Initialize the Vision API client
            client = vision.ImageAnnotatorClient()

            # Load the image
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)

            # Perform object detection
            response = client.object_localization(image=image)
            objects = response.localized_object_annotations

            # Parse detected objects
            detected_objects = [obj.name for obj in objects]

            if not detected_objects:
                return ["No objects detected"]
            
            return detected_objects
        except Exception as e:
            raise Exception(f"Error in object detection: {str(e)}")

    @staticmethod
    def extract_frames(video_path: str, output_dir: str = "frames", interval: int = 30):
        """
        Extract frames from a video at regular intervals.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                # Save the frame as an image file
                frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_frames.append(frame_path)

            frame_count += 1

        cap.release()
        return saved_frames

class SpeechToText:
    @staticmethod
    def transcribe_audio(audio_path: str):
        """
        Transcribe the audio file into text using Google Cloud Speech-to-Text API.
        """
        try:
            client = speech.SpeechClient()

            # Load the audio file
            with io.open(audio_path, "rb") as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=48000,
                language_code="en-US",
            )

            # Perform the transcription
            response = client.recognize(config=config, audio=audio)
            transcript = " ".join(result.alternatives[0].transcript for result in response.results)

            return transcript
        except Exception as e:
            raise Exception(f"Error transcribing audio: {str(e)}")

class VideoAgent:
    def process_video_query(self, video_file: UploadFile):
        """
        Process the uploaded video file and respond based on detected objects and audio transcription.
        """
        try:
            # Save the uploaded video file
            video_path = f"temp_{video_file.filename}"
            audio_path = "temp_audio.wav"
            with open(video_path, "wb") as f:
                f.write(video_file.file.read())

            # Extract frames for object detection
            frames = ObjectDetection.extract_frames(video_path)

            # Perform object detection on each frame
            all_detected_objects = []
            for frame in frames:
                detected_objects = ObjectDetection.detect_objects(frame)
                all_detected_objects.extend(detected_objects)

            # Remove duplicate objects
            unique_objects = list(set(all_detected_objects))

            # Extract audio from the video
            AudioProcessor.extract_audio(video_path, audio_path)

            # Transcribe audio to text
            user_query = SpeechToText.transcribe_audio(audio_path)

            print(f"Detected objects: {unique_objects}")
            print(f"Transcribed query: {user_query}")

            # Combine detected objects and user query
            llm_prompt = (
                f"The user showed the following objects in the video: {', '.join(unique_objects)}. "
                f"The user also asked: '{user_query}'. "
                "Answer the user's query based on these inputs."
            )

            # Query LLM
            llm_response = llm.invoke(llm_prompt)  # Assuming `llm` is ChatOpenAI or similar

            return llm_response
        except Exception as e:
            raise Exception(f"Error processing video query: {str(e)}")
