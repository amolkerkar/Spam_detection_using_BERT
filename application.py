from flask import Flask, render_template, request, redirect, url_for
from backend import *
app = Flask(__name__)

# Set the path to the uploads folder
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    # Read the uploaded file directly without saving it
    content = file.read().decode('utf-8').splitlines()

    new_content = [val[1:-2] for val in content]


    classified_content = classifier(new_content)

    # Render the response template with modified content
    return render_template('response.html', content = classified_content)


if __name__ == '__main__':
    app.run(debug=True)
