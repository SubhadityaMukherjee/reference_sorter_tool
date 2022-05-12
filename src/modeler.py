#!/usr/bin/python3
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


# %%
image_size = (200, 200)
batch_size = 128
main_directory = Path("/media/hdd/Datasets/asl")
keras.mixed_precision.set_global_policy("mixed_float16")
# %%
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    str(main_directory / "asl_alphabet_train"),
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    str(main_directory / "asl_alphabet_train"),
    subset="validation",
    validation_split=0.2,
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
# %%
train_ds
# %%

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
# %%
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)
# %%
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
# %%
train_ds = train_ds.prefetch(buffer_size=batch_size)
val_ds = val_ds.prefetch(buffer_size=batch_size)
# %%


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


# %%
model = make_model(input_shape=image_size + (3,), num_classes=30)
keras.utils.plot_model(model, show_shapes=True)
# %%
epochs = 1

callbacks = [
    keras.callbacks.ModelCheckpoint("./logs/save_at_{epoch}.h5"),
    keras.callbacks.ProgbarLogger(count_mode="samples", stateful_metrics=None),
]
loss_fn = keras.losses.SparseCategoricalCrossentropy()
opt = keras.optimizers.Adam(1e-3)

model.compile(
    optimizer=opt,
    loss=loss_fn,
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

# %%
img = keras.preprocessing.image.load_img(
    "/media/hdd/Datasets/asl/asl_alphabet_test/B_test.jpg", target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array).argmax(axis=-1)
score = predictions[0]
print(score)

# %%
