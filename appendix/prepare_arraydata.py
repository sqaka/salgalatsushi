from PIL import Image
import glob
import numpy as np

LABELS = ['monkey', 'atsushi', 'onna', 'others']
NUM_LABELS = len(LABELS)
IMAGE_SIZE = 128
NUM_MAXDATA = 144
NUM_TESTDATA = NUM_LABELS * 5

x_train = []
x_test = []
y_train = []
y_test = []

for index, label in enumerate(LABELS):
    image_dir = './images/' + label
    files = glob.glob(image_dir + '/*')
    for i, file in enumerate(files):
        if i >= NUM_MAXDATA:
            break
        image = Image.open(file)
        image = image.convert('RGB')
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        data = np.asarray(image)
        if i < NUM_TESTDATA:
            x_test.append(data)
            y_test.append(index)
        else:
            for angle in range(-20, 20, 5):
                # incline
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                x_train.append(data)
                y_train.append(index)

                # transpose
                img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                x_train.append(data)
                y_train.append(index)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

xy = (x_train, x_test, y_train, y_test)
np.save('./predata_{}.npy'.format(IMAGE_SIZE), xy)
