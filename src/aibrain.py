# %%
from pathlib import Path
import os
import io

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Activation,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import applications


# %%
class MainModel:
    def __init__(self, folder, epochs):
        self.input_shape = (128, 128, 3)
        self.num_classes = len(os.listdir(folder))
        self.epochs = epochs
        self.model = None

    def create_model(self, model_name="efficientnet-b1"):
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

    def get_model_summary(self):
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + "\n"))
        summary_string = stream.getvalue()
        stream.close()
        return summary_string
