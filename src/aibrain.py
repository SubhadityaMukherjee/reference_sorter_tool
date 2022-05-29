# %%
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from tkinter import Image
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# import PySimpleGUI as sg
from PIL import Image, ImageTk  # Image for open, ImageTk for display
from tensorflow import keras
from tensorflow.keras import applications, layers
from tensorflow.keras.layers import (Activation, Dense, Dropout, Flatten,
                                     GlobalAveragePooling2D)
from tensorflow.keras.models import Model, Sequential

from .utils import *

# %%


class MainModel:
    def __init__(self, folder: str, mxp: bool):
        self.folder = folder
        self.image_size: tuple = (128, 128)
        self.num_classes: int = len(os.listdir(folder))
        self.classes = os.listdir(folder)
        self.classes.remove("unlabeled")
        # self.epochs = epochs
        self.mxp = mxp
        self.autotune = tf.data.experimental.AUTOTUNE
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy()
        self.opt = keras.optimizers.Adam(1e-3)
        self.create_model()

    def create_model(self, model_name="EfficientNet"):
        if model_name == "EfficientNet":
            base_model = keras.applications.EfficientNetB1(
                weights="imagenet", include_top=False
            )
        if model_name == "Resnet":
            base_model = keras.applications.ResNet50(
                weights="imagenet", include_top=False
            )
        if model_name == "Vgg16":
            base_model = keras.applications.VGG16(weights="imagenet", include_top=False)
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # add a fully-connected layer
        x = Dense(512, activation="relu")(x)
        # and a fully connected output/classification layer
        predictions = Dense(self.num_classes, activation="softmax")(x)
        # create the full network so we can train on it
        self.model = Model(inputs=base_model.input, outputs=predictions)
        callbacks = [
            keras.callbacks.ModelCheckpoint("./logs/save_at_{epoch}.h5"),
            keras.callbacks.ProgbarLogger(count_mode="samples", stateful_metrics=None),
        ]
        self.model.compile(
            optimizer=self.opt,
            loss=self.loss_fn,
            metrics=["accuracy"],
        )

    def get_model_summary(self):
        summary_string = ""
        if self.model != None:
            from keras.utils.layer_utils import count_params

            summary_string += "Model Summary:\n"
            summary_string += "Number of parameters: {}\n".format(
                count_params(self.model.trainable_weights)
            )

        return summary_string

    def check_classes(self):
        impaths, _, classes = load_images_from_folder(self.folder)
        countunique = np.unique(classes, return_counts=True)
        if any([x for x in countunique if x < 10]) == True:
            sg.Popup("Not enough examples of each class, please run the labeler")

    def openwindow(self, classes, curpath):
        layout = [
            [
                sg.Image(key="-IMAGE-"),
                # sg.DropDown(classes, size=(20, 1), key="_LIST_"),
                [sg.Combo(classes, enable_events=False, key="combo")],
                # sg.Listbox(classes, size=(20,4), enable_events=True, key='_LIST_'),
                # sg.Button("Next"),
                sg.Exit(),
            ],
        ]

        window = sg.Window("labs", layout)
        event, values = window.read()
        im = Image.open(curpath)
        if event == sg.WIN_CLOSED or event == "Exit":
            window["-IMAGE-"].update(data=ImageTk.PhotoImage(im), size=(128, 128))
            time.sleep(1)
            return values["combo"], window

    def labeler(self, folder):
        impaths, _, classes = load_images_from_folder(folder)
        impaths = iter(impaths)
        while True:
            # uni, counts = np.unique(classes, return_counts=True)
            # countunique = np.asarray((uni, counts)).T
            countunique = Counter(classes).values()
            if any([x for x in countunique if int(x) < 10]) == True:
                curpath = next(impaths)
                while True:
                    wins, window = self.openwindow(classes, curpath)
                    print(wins)
                    if len(wins) > 2:
                        wins.close()

    def data_setup(self):
        try:
            train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                str(self.folder),
                validation_split=0.2,
                subset="training",
                seed=1337,
                image_size=self.image_size,
                batch_size=self.autotune,
                # classes = self.classes,
            )
            val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                str(self.folder),
                subset="validation",
                validation_split=0.2,
                seed=1337,
                image_size=self.image_size,
                batch_size=self.autotune,
                # classes = self.classes,
            )
            return train_ds, val_ds
        except Exception as e:
            print(e)
            # print("Not enough classes in data. Please run the labeler")
            return None, None

    def train_model(self, epochs):
        train_ds, val_ds = self.data_setup()
        if train_ds is None or val_ds is None:
            sg.Popup("Not enough classes in data. Please run the labeler")
        else:
            progwindow = progress_bar(0, epochs, "Running Training")
            for epoch in range(epochs):
                event, values = progwindow.read(timeout=0)
                for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
                    with tf.GradientTape() as tape:
                        logits = self.model(x_batch_train, training=True)
                        loss_value = self.loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.opt.apply_gradients(zip(grads, self.model.trainable_weights))
                progwindow["progbar"].update(epoch)
            progwindow.close()
