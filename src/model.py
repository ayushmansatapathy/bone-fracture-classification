import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = (224, 224)
NUM_CLASSES = 10  # You have 10 fracture types

def build_model():
    """
    Builds a transfer learning model using MobileNetV2
    """

    # Load pretrained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

# Freeze most layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Unfreeze last 30 layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

    # Custom classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    # Final model
    model = models.Model(inputs=base_model.input, outputs=outputs)

    return model