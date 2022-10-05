from flask import render_template
from flask import Flask, request
import os

from models.classifier import *
UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('master.html')


# web page that handles user image and displays model results
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    file = request.files['file']
    if file and allowed_file(file.filename):
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)

        # predict function
        prediction = dog_breed_detector(img_path)
        # delete uploaded file
        os.remove(img_path)
        return render_template(
                    'go.html',
                    prediction=prediction
                )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
