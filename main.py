import numpy as np
import pandas as pd

import os
import warnings

import tensorflow
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')

from pathlib import Path

#Visualisation Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix

#Keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, AvgPool2D, Input, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# setup GPU
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    tensorflow.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tensorflow.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

#Image Transformation Libraries
import cv2
from imgaug import augmenters as iaa

# Let's start with the hyperparameters
base_learning_rate = 1e-5
batch_size=32
epochs = 1

plot_dir='plot_' + str(batch_size) + '_' + str(epochs)
model_dir='model_' + str(batch_size) + '_' + str(epochs)
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

print("plot_dir: ", plot_dir)
print("model_dir: ", model_dir)

print(os.listdir("data/chest_xray/"))

base_dir = "data/chest_xray/"
train_dir = base_dir+'train/'
test_dir = base_dir+'test/'
val_dir = base_dir+'val/'

def get_df(path):
    lst = []
    normal_dir = Path(path + "NORMAL")
    pneumonia_dir = Path(path + "PNEUMONIA")
    normal_data = normal_dir.glob("*.jpeg")
    pneumonia_data = pneumonia_dir.glob("*.jpeg")
    for fname in normal_data:
        lst.append((fname, 0))
    for fname in pneumonia_data:
        lst.append((fname, 1))
    df = pd.DataFrame(lst, columns=['Image', 'Label'], index=None)
    s = np.arange(df.shape[0])
    np.random.shuffle(s)
    df = df.iloc[s,:].reset_index(drop=True)
    return df

df_train = get_df(train_dir)
df_val = get_df(val_dir)
df_test = get_df(test_dir)

print(df_train.shape)
print(df_val.shape)
print(df_test.shape)

fig, ax = plt.subplots(figsize=(8, 8))
sns.countplot(df_train['Label'])
ax.set_title('Distribution of Images', fontsize=14)
ax.set_xlabel('Label', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.savefig(plot_dir + "/distribution_of_images.png")

def transform_image(img_list):
    img = cv2.resize(img_list, (224, 224))
    #cv2 reads image in BGR format. Let's convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        img_list.append(transform_image(cv2.imread(str(img_path_list[i]))))
    n = len(img_list)
    if is_augment:
        for i in range(n):
            img = img_list[i]
            img = augment_image(img)
            img_list.append(img)
        img_list = np.array(img_list)
        label_list = np.append(label_list, label_list)
    return img_list, label_list


def generate_batch_images(df, batch_size):
    s = np.arange(df.shape[0])
    np.random.shuffle(s)
    X_dev = np.array(df_train.iloc[s, 0])
    Y_dev = np.array(df_train.iloc[s, 1])
    start_index = 0
    while start_index < len(X_dev):
        if start_index + batch_size <= len(X_dev):
            end_index = start_index + batch_size
        else:
            end_index = len(X_dev)
        # Select image paths in batches
        x_dev = X_dev[start_index:end_index]
        y_dev = Y_dev[start_index:end_index]

        # Transform images and augment
        x_dev, y_dev = transform_augment_batch(x_dev, y_dev, True)
        y_dev = y_dev.reshape((len(y_dev), 1))

        # Normalize
        x_dev = x_dev / 255.0
        yield x_dev, y_dev

fig, ax = plt.subplots(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = transform_image(cv2.imread(str(df_train.iloc[i, 0])))
    plt.imshow(img)
    if df_train.iloc[i, 1] == 0:
        plt.title('Normal')
    else:
        plt.title('Pneumonia')
    plt.xticks([])
    plt.yticks([])
plt.savefig(plot_dir + "/transform_image.png")

plt.subplots(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i+1)
    img = transform_image(cv2.imread(str(df_train.iloc[i, 0])))
    img = augment_image(img)
    plt.imshow(img)
    if df_train.iloc[i, 1] == 0:
        plt.title('Normal')
    else:
        plt.title('Pneumonia')
    plt.xticks([])
    plt.yticks([])
plt.savefig(plot_dir + "/augment_image.png")

val_labels = np.array(df_val.iloc[:, 1]).reshape((df_val.shape[0], 1))
val_images, _ = transform_augment_batch(df_val.iloc[:, 0], df_val.iloc[:, 1], False)
val_images = np.array(val_images)
val_images = val_images / 255.0
print(val_images.shape, val_labels.shape)

test_labels = np.array(df_test.iloc[:, 1]).reshape((df_test.shape[0], 1))
test_images, _ = transform_augment_batch(df_test.iloc[:, 0], df_test.iloc[:, 1], False)
test_images = np.array(test_images)
test_images = test_images / 255.0
print(test_images.shape, test_labels.shape)

def create_model():
    img_input = Input(shape=(224, 224, 3))

    # Block 1
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='block1_conv1')(
        img_input)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='block1_conv2')(
        x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='block1_pool')(x)

    # Block 2
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='block2_conv1',
               trainable=False)(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='block2_conv2',
               trainable=False)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='block2_pool')(x)

    # Block 3
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='block3_conv1',
               trainable=False)(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='block3_conv2',
               trainable=False)(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='block3_conv3',
               trainable=False)(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='block3_conv4',
               trainable=False)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='block3_pool')(x)

    # Block 4
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block4_conv1')(x)
    x = BatchNormalization(name='block4_bn1')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block4_conv2')(x)
    x = BatchNormalization(name='block4_bn2')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block4_conv3')(x)
    x = BatchNormalization(name='block4_bn3')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block4_conv4')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='block4_pool')(x)

    # Block 5
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block5_conv1')(x)
    x = BatchNormalization(name='block5_bn1')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block5_conv2')(x)
    x = BatchNormalization(name='block5_bn2')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block5_conv3')(x)
    x = BatchNormalization(name='block5_bn3')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block5_conv4')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='block5_pool')(x)

    # Other layers
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.7, name='dropout_1')(x)
    x = Dense(1000, activation='relu', name='fc2')(x)
    x = Dropout(0.7, name='dropout_2')(x)
    x = Dense(512, activation='relu', name='fc3')(x)
    x = Dropout(0.5, name='dropout_3')(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=img_input, outputs=x)
    return model

model = create_model()
print(model.summary())

adam = Adam(lr = base_learning_rate)
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=adam)

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',
                                patience=3,
                                verbose=1,
                                factor=0.2,
                                min_lr=1e-7)

model_chkpoint = ModelCheckpoint(filepath=model_dir + '/vgg_19_model.hdf5', save_best_only=True, save_weights_only=True)
data_generator = generate_batch_images(df_train, batch_size)

history = model.fit_generator(data_generator, epochs=epochs, steps_per_epoch=df_train.shape[0]/batch_size,
                    callbacks=[reduce_lr, model_chkpoint], validation_data=(val_images, val_labels),
                    class_weight={0:3, 1:1})

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(plot_dir + '/' + 'train_proc.png')

# serialize model to JSON
model_json = model.to_json()
with open(model_dir + "/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(model_dir + "/model.h5")
print("Saved model to disk")

test_logits = model.predict(test_images, batch_size=16)

# Evaluation on test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=16)
print("Loss on test set: ", test_loss)
print("Accuracy on test set: ", test_acc)

cm  = confusion_matrix(test_labels, np.round(test_logits))
plt.figure()
plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True, cmap=plt.cm.Oranges)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.savefig(plot_dir + "/confusion_matrix.png")

true_negative, false_positive, false_negative, true_positive  = cm.ravel()
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)

print('Precison of chest X-ray for pneumonia:{:.2f}'.format(precision))
print('Recall of chest X-ray for pneumonia:{:.2f}'.format(recall))