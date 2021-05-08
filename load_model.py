from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, AvgPool2D, Input, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential, Model, model_from_json
from data_load import DataLoader
import numpy as np
import cv2 as cv
from imgaug import augmenters as iaa
from tensorflow.keras.optimizers import Adam

def transform_image(img_list):
    img = cv.resize(img_list, (224, 224))
    #cv reads image in BGR format. Let's convert to RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

def augment_image(img_list):
    seq = iaa.OneOf([
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            rotate=(-25, 25)
        ),
        iaa.Fliplr(),
        iaa.Multiply((1.2, 1.5))
    ])
    return seq.augment_image(img_list)

def transform_augment_batch(img_path_list, label_list, is_augment=False):
    img_list = []
    for i in range(len(img_path_list)):
        img_list.append(transform_image(cv.imread(str(img_path_list[i]))))
    n = len(img_list)
    if is_augment:
        for i in range(n):
            img = img_list[i]
            img = augment_image(img)
            img_list.append(img)
        img_list = np.array(img_list)
        label_list = np.append(label_list, label_list)
    return img_list, label_list

if __name__ == "__main__":
    # load json and create model
    json_file = open('model_32_8/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_32_8/model.h5")
    print("Loaded model from disk")

    dloader = DataLoader('data/chest_xray/')
    test_data = dloader.get_test_data()

    test_labels = np.array(test_data.iloc[:, 1]).reshape((test_data.shape[0], 1))
    test_images, _ = transform_augment_batch(test_data.iloc[:, 0], test_data.iloc[:, 1], False)
    test_images = np.array(test_images)
    test_images = test_images / 255.0
    print(test_images.shape, test_labels.shape)

    # evaluate loaded model on test data
    adam = Adam(lr=1e-5)
    loaded_model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=adam)
    out = loaded_model.predict(test_images, batch_size=16)
    # print("Model output: ", out)