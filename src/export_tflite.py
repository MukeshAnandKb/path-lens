# src/export_tflite.py
import tensorflow as tf
import numpy as np
import os
from src.dataset import prepare_train_val

SAVED_MODEL_DIR = 'saved_models/final_model'
TFLITE_OUT = 'saved_models/model_quant.tflite'

def representative_gen(train_ds, max_samples=500):
    # train_ds yields ((image_batch, sensor_batch), label_batch)
    count = 0
    for (img_b, sens_b), _ in train_ds.unbatch().batch(1):
        img = img_b.numpy().astype('float32')
        sens = sens_b.numpy().astype('float32')
        yield [img, sens]
        count += 1
        if count >= max_samples:
            break

def convert():
    # prepare small dataset to use as representative (load from CSV)
    train_ds, _, _, _, _ = prepare_train_val(csv_path='data/sensors.csv', images_dir='data/images',
                                             batch_size=8, val_split=0.15, save_dir='saved_models')
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_gen(train_ds, max_samples=300)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(TFLITE_OUT, 'wb') as f:
        f.write(tflite_model)
    print("Wrote TFLite model to:", TFLITE_OUT)

if __name__ == '__main__':
    convert()
