# src/fusion_model.py
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_cane_fusion_model(img_shape=(224,224,3),
                            sensor_dim=5,
                            vision_feat_dim=128,
                            sensor_feat_dim=32,
                            num_classes=4,
                            freeze_backbone=True):
    # Vision backbone
    base = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                             include_top=False,
                                             weights='imagenet')
    if freeze_backbone:
        base.trainable = False

    x = layers.GlobalAveragePooling2D(name='vision_gap')(base.output)
    x = layers.Dense(vision_feat_dim, activation='relu', name='vision_proj')(x)

    # Sensor branch
    s_in = layers.Input(shape=(sensor_dim,), name='sensor_input')
    s = layers.Dense(64, activation='relu')(s_in)
    s = layers.BatchNormalization()(s)
    s = layers.Dense(sensor_feat_dim, activation='relu', name='sensor_feat')(s)

    # Fusion
    fused = layers.Concatenate(name='fusion_concat')([x, s])
    f = layers.Dense(128, activation='relu')(fused)
    f = layers.Dropout(0.25)(f)
    f = layers.Dense(64, activation='relu')(f)
    outputs = layers.Dense(num_classes, activation='softmax', name='pred')(f)

    model = Model(inputs=[base.input, s_in], outputs=outputs, name='cane_fusion_model')
    return model
