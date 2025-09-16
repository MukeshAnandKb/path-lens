# src/eval.py
import numpy as np
import tensorflow as tf
import json, os
from src.dataset import _preprocess_image
from PIL import Image

TFLITE_PATH = 'saved_models/model_quant.tflite'
LABEL_MAP_PATH = 'saved_models/label_map.json'
SENSOR_STATS_PATH = 'saved_models/sensor_stats.npz'

def load_label_map():
    with open(LABEL_MAP_PATH, 'r') as f:
        return json.load(f)

def normalize_sensor(sensor, stats_path=SENSOR_STATS_PATH):
    arr = np.load(stats_path)
    mean = arr['mean']
    std = arr['std']
    return (np.array(sensor, dtype=np.float32) - mean) / std

def run_one(image_path, sensor_vector):
    label_map = load_label_map()
    inv_map = {v:k for k,v in label_map.items()}
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # preprocess image
    img = _preprocess_image(image_path).numpy().astype('float32')  # shape (224,224,3)
    # add batch
    img_in = np.expand_dims(img, axis=0)

    sens_norm = normalize_sensor(sensor_vector)
    sens_in = np.expand_dims(sens_norm.astype('float32'), axis=0)

    # handle quantization
    for i, inp in enumerate(input_details):
        if inp['dtype'] == np.int8 or inp['dtype'] == np.uint8:
            scale, zero_point = inp['quantization']
            if inp['shape'][1:] == (224,224,3):  # image input
                q = (img_in / scale + zero_point).astype(np.int8)
                interpreter.set_tensor(inp['index'], q)
            else:  # sensor input
                q = (sens_in / scale + zero_point).astype(np.int8)
                interpreter.set_tensor(inp['index'], q)
        else:
            # float
            if inp['shape'][1:] == (224,224,3):
                interpreter.set_tensor(inp['index'], img_in)
            else:
                interpreter.set_tensor(inp['index'], sens_in)

    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    # if output is quantized
    if output_details[0]['dtype'] in (np.int8, np.uint8):
        scale, zero_point = output_details[0]['quantization']
        out = scale * (out.astype(np.float32) - zero_point)
    pred = np.argmax(out, axis=1)[0]
    print("Predicted label:", inv_map[pred], "softmax:", out)

if __name__ == '__main__':
    # example usage: replace with a real image filename and sensor values
    img_path = 'data/images/example.jpg'   # put a sample image here
    sensor_vector = [0.0, 0.0, 0.0, 0.1, 0.05]  # sample: tilt_x, tilt_y, tilt_z, moisture, piezo
    run_one(img_path, sensor_vector)
