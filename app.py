import numpy as np
import os
from PIL import Image

from flask import Flask, render_template, request, redirect
from flask import url_for, abort, flash, send_from_directory
from werkzeug.utils import secure_filename
from keras.models import load_model

from utils.eval import calc_percentage, evaluation

UPLOAD_FOLDER = './static/images/original'
EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL = './static/models/apexile_128.h5'
# MODEL = './static/models/squeeze50_loss1.18_acc0.77.h5'
LABELS = ['monkey', 'atsushi', 'onna', 'others']
NUM_LABELS = len(LABELS)
IMAGE_SIZE = 128


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in EXTENSIONS


def image_to_array(target_image_path):
    image = Image.open(target_image_path)
    image = image.convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_asarray = np.asarray(image)
    data = []
    data.append(image_asarray)
    array_data = np.array(data)
    return array_data


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('エラー１：ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('エラー２：ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            model = load_model(MODEL)
            face_detect_image_path, target_image_path = evaluation(file_path)
            if target_image_path is None:
                return render_template('index.html')

            array_data = image_to_array(target_image_path)
            percentages = calc_percentage(array_data, model)
            results = [percentages, face_detect_image_path, target_image_path]

            return render_template('result.html', results=results)
    return render_template('index.html')


@ app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def main():
    app.debug = True
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


if __name__ == '__main__':
    main()
