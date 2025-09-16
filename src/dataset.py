# src/dataset.py
import os, json
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = (224,224)
SENSOR_COLS = ['tilt_x','tilt_y','tilt_z','moisture','piezo']

def _preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    # Use MobileNetV2 preprocessing: scales to [-1,1]
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def build_label_map(df, save_dir='saved_models'):
    labels = sorted(df['label'].unique())
    label_map = {lab: i for i, lab in enumerate(labels)}
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f)
    return label_map

def compute_sensor_stats(df, save_dir='saved_models'):
    stats = {}
    sensors = df[SENSOR_COLS].astype(float)
    stats['mean'] = sensors.mean().to_dict()
    stats['std'] = sensors.std().replace(0,1).to_dict()
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, 'sensor_stats.npz'),
             mean=np.array([stats['mean'][c] for c in SENSOR_COLS]),
             std=np.array([stats['std'][c] for c in SENSOR_COLS]))
    return stats

def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        raise ValueError("CSV must have a 'label' column.")
    return df

def create_tf_dataset_from_df(df, images_dir, label_map, sensor_mean, sensor_std,
                              batch_size=32, shuffle=True):
    filenames = (images_dir + "/" + df['filename']).tolist()
    sensors = df[SENSOR_COLS].fillna(0).astype('float32').values
    # normalize sensors using mean/std
    sensors = (sensors - sensor_mean) / sensor_std
    labels = df['label'].map(label_map).astype('int').values

    ds_img = tf.data.Dataset.from_tensor_slices(filenames)
    ds_img = ds_img.map(lambda p: _preprocess_image(p), num_parallel_calls=tf.data.AUTOTUNE)
    ds_sens = tf.data.Dataset.from_tensor_slices(sensors)
    ds_lbl = tf.data.Dataset.from_tensor_slices(labels)

    ds = tf.data.Dataset.zip(((ds_img, ds_sens), ds_lbl))
    if shuffle:
        ds = ds.shuffle(buffer_size=2048)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def prepare_train_val(csv_path='data/sensors.csv', images_dir='data/images',
                      batch_size=32, val_split=0.2, save_dir='saved_models'):
    df = load_dataframe(csv_path)
    label_map = build_label_map(df, save_dir)
    stats = compute_sensor_stats(df, save_dir)
    # split
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=42, stratify=df['label'])
    sensor_mean = np.array([stats['mean'][c] for c in SENSOR_COLS], dtype='float32')
    sensor_std = np.array([stats['std'][c] for c in SENSOR_COLS], dtype='float32')
    train_ds = create_tf_dataset_from_df(train_df, images_dir, label_map, sensor_mean, sensor_std, batch_size, shuffle=True)
    val_ds = create_tf_dataset_from_df(val_df, images_dir, label_map, sensor_mean, sensor_std, batch_size, shuffle=False)
    return train_ds, val_ds, label_map, sensor_mean, sensor_std

om murugan manraj
