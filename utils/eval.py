#! -*- coding: utf-8 -*-
import datetime
import cv2


IMAGE_PATH = './static/images/face_detect/'
FACE_PATH = './static/images/cut_face/'
CASCADE_PATH = './static/models/haarcascade_frontalface_default.xml'
EVAL_RESULT = {
    0: u"猿",
    1: u"ギャル",
    2: u"EXILE TRIVE",
    3: u'だれ…？'
}


def evaluation(file_path):
    now = datetime.datetime.now()
    timestamp = now.strftime('%m%d%H%M%S')
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(CASCADE_PATH)
    face = faceCascade.detectMultiScale(gray, 1.1, 3)
    if len(face) > 0:
        for rect in face:
            cv2.rectangle(img, tuple(rect[0:2]), tuple(
                rect[0:2]+rect[2:4]), (0, 0, 255), thickness=2)
            face_detect_image_path = '{}{}.jpg'.format(IMAGE_PATH, timestamp)

            cv2.imwrite(face_detect_image_path, img)
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]

            target_image_path = '{}{}.jpg'.format(FACE_PATH, timestamp)
            cv2.imwrite(target_image_path, img[y:y+h, x:x+w])
            return face_detect_image_path, target_image_path

    else:
        return None


def calc_percentage(array_data, model):
    ndarrays = model.predict(array_data)
    result_data = ndarrays[0]
    result = [round(n * 100.0, 1) for n in result_data]
    labels = []

    for index, rate in enumerate(result):
        name = EVAL_RESULT[index]
        labels.append({
            'label': index,
            'name': name,
            'rate': rate
        })
    percentages = sorted(labels, key=lambda x: x['rate'], reverse=True)

    return percentages
