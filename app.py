from flask import Flask, request, render_template, redirect, url_for, flash, Response
import os
import cv2

# initialize flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.secret_key = 'supersecretkey'

# ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('output_frames', exist_ok=True)

# upload video
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No video part')
            return redirect(request.url)

        video_file = request.files['video']

        if video_file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_path)
        flash(f'Video {video_file.filename} uploaded successfully!')

        preprocess_video(video_path)

        return redirect(url_for('home'))

    return render_template('index.html')

# real time capture
@app.route('/capture')
def capture_video():
    return Response(capture_real_time_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

def capture_real_time_video():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

def preprocess_video(video_path, output_dir='output_frames', frame_rate=1):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            resized_frame = cv2.resize(frame, (224, 224))
            frame_path = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_path, resized_frame)
            saved_count += 1

        frame_count += 1

    cap.release()

if __name__ == '__main__':
    app.run(debug=True)