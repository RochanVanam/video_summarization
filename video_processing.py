import openai
import cv2
import os
import whisper
import subprocess
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv(verbose=True, override=True)

openai.api_key = os.getenv('OPENAI_API_KEY')

whisper_model = whisper.load_model("base")  # for audio
yolo_model = YOLO("yolov8s.pt") # for video

# Extract audio
def extract_audio_from_video(video_path, output_audio_path='audio.wav'):
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return False

    try:
        # Use FFmpeg to extract audio
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", output_audio_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if os.path.exists(output_audio_path):
            print(f"Audio extracted successfully to {output_audio_path}")
            return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e.stderr.decode()}")
        return False

    return False

# Transcribe audio
def transcribe_audio(audio_path='audio.wav'):
    result = whisper_model.transcribe(audio_path)
    return result['text']

def detect_objects_in_frame(frame):
    """
    Detect objects in a given video frame using YOLOv8.

    Args:
        frame (numpy.ndarray): Input video frame.

    Returns:
        list: List of detected object names.
    """
    # Perform inference on the frame
    results = yolo_model(frame)

    # Extract detected object names
    object_names = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)  # Class ID
            label = yolo_model.names[cls_id]  # Map ID to label
            object_names.append(label)

    return object_names

# Generate summary
def generate_gpt4_summary(audio_transcription, object_summary):
    prompt = f"""
    You are a summarization assistant. Below is a transcript of a video, along with a summary of key objects detected in the video frames.

    Audio Transcript:
    {audio_transcription}

    Key Objects Detected:
    {object_summary}

    Please provide a short, coherent text summary of the video, based on the above information.
    """

    print(prompt)

    response = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        stream=True
    )
    responses = ''
    for chunk in response:
        response_content = chunk.choices[0].delta.content
        if response_content:
            responses += response_content

    return responses.strip()

# Process video for summary
def process_video_for_summary(video_path):
    # Extract audio and transcribe
    audio_output_path = "static/uploads/audio.wav"
    extract_audio_from_video(video_path, audio_output_path)
    audio_transcription = transcribe_audio(audio_output_path)

    # Detect objects
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    object_summary = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 30 == 0:  # every 30th frame
            objects = detect_objects_in_frame(frame)
            object_summary.append(f"Frame {frame_count}: {', '.join(objects)}")

        frame_count += 1

    cap.release()

    # Generate summary
    summary = generate_gpt4_summary(audio_transcription, "\n".join(object_summary))

    if os.path.exists(video_path):
        os.remove(video_path)
        print(f"Deleted uploaded video: {video_path}")

    if os.path.exists(audio_output_path):
        os.remove(audio_output_path)
        print(f"Deleted extracted audio: {audio_output_path}")

    return summary
