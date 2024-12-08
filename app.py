from flask import Flask, request, render_template, redirect, url_for, flash
import os
from video_processing import process_video_for_summary

# initialize flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.secret_key = 'supersecretkey'

# ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Upload video
@app.route('/', methods=['GET', 'POST'])
def home():
    summary = None

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

        # Process video for summarization
        summary = process_video_for_summary(video_path)

    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
