# src/train.py
import os
import tensorflow as tf
from src.dataset import prepare_train_val
from src.fusion_model import build_cane_fusion_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

def main():
    os.makedirs('saved_models', exist_ok=True)
    train_ds, val_ds, label_map, sensor_mean, sensor_std = prepare_train_val(csv_path='data/sensors.csv',
                                                                            images_dir='data/images',
                                                                            batch_size=16,
                                                                            val_split=0.15,
                                                                            save_dir='saved_models')

    num_classes = len(label_map)
    model = build_cane_fusion_model(num_classes=num_classes, freeze_backbone=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    callbacks = [
        ModelCheckpoint('saved_models/best_model.h5', save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=8, restore_best_weights=True),
        TensorBoard(log_dir='logs')
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks)
    # Save full TF SavedModel (needed for TFLite conversion)
    model.save('saved_models/final_model', include_optimizer=False)
    print("Saved model to saved_models/final_model")

if __name__ == '__main__':
    main()
