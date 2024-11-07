import openai
import cv2
import os
import whisper
from yolov5 import YOLOv5
from dotenv import load_dotenv

load_dotenv(verbose=True, override=True)

openai.api_key = os.getenv('OPENAI_API_KEY')

whisper_model = whisper.load_model("base")  # for audio

# extract audio
def extract_audio_from_video(video_path, output_audio_path='audio.wav'):
    video_capture = cv2.VideoCapture(video_path)
    audio_capture = cv2.VideoCapture(video_path)
    audio_capture.set(cv2.CAP_PROP_AUDIO_STREAM, 1)

    if not audio_capture.isOpened():
        print(f"Error extracting audio from {video_path}")
        return

    os.system(f"ffmpeg -i {video_path} -vn {output_audio_path}")

# transcribe audio
def transcribe_audio(audio_path='audio.wav'):
    result = whisper_model.transcribe(audio_path)
    return result['text']

# detect objects
def detect_objects_in_frame(frame):
    # need to train model for object detection

# generate summary
def generate_gpt4_summary(audio_transcription, object_summary):
    prompt = f"""
    You are a summarization assistant. Below is a transcript of a video, along with a summary of key objects detected in the video frames.

    Audio Transcript:
    {audio_transcription}

    Key Objects Detected:
    {object_summary}

    Please provide a short, coherent text summary of the video, based on the above information.
    """

    response = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': prompt},
        ],
        temperature=0,
        stream=True
    )
    responses = ''
    for chunk in response:
        response_content = chunk.choices[0].delta.content
        if response_content:
            responses += response_content

    return responses.strip()

def process_video_for_summary(video_path):
    # extract audio and transcribe
    extract_audio_from_video(video_path)
    audio_transcription = transcribe_audio()

    # detect objects
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    object_summary = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 30 == 0:  # every 30th frame
            objects = detect_objects_in_frame(frame)
            object_summary.append(f"Objects detected in frame {frame_count}: {', '.join(objects)}")

        frame_count += 1

    cap.release()

    # generate summary
    summary = generate_gpt4_summary(audio_transcription, "\n".join(object_summary))
    return summary
