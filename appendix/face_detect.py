import cv2
import glob

LABEL = 'label_name'
INPUT_DATA_PATH = './{}/'.format(LABEL)
SAVE_PATH = './{}_face/'.format(LABEL)
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(CASCADE_PATH)
files = glob.glob('{}*.jpg'.format(INPUT_DATA_PATH))
face_detect_count = 0

for fname in files:
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(gray, 1.1, 3)

    if len(face) > 0:
        for rect in face:
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]

            cv2.imwrite('{}{}{}.jpg'.format(
                SAVE_PATH, LABEL, face_detect_count), img[y:y+h, x:x+w])
            face_detect_count += 1

    else:
        print(fname + ':NoFace')
