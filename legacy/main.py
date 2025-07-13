import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.models import Model
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import math

dataset_dir = "/home/siton01/watermelon_eval/datasets"

subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

datasets = []

for subdir in subdirs:

    data_id, label = subdir.split("_")
    label = float(label)

    folders = [f for f in os.listdir(os.path.join(dataset_dir, subdir, "chu")) if
               os.path.isdir(os.path.join(dataset_dir, subdir, "chu", f))]

    wav_data_list = []
    jpg_files = []
    labels = []

    for folder in folders:
        wav_file = os.path.join(dataset_dir, subdir, "chu", folder,
                                [f for f in os.listdir(os.path.join(dataset_dir, subdir, "chu", folder)) if
                                 f.endswith(".wav")][0])

        jpg_file = os.path.join(dataset_dir, subdir, "chu", folder,
                                [f for f in os.listdir(os.path.join(dataset_dir, subdir, "chu", folder)) if
                                 f.endswith(".jpg")][0])

        audio = tf.io.read_file(wav_file)
        audio, sample_rate = tf.audio.decode_wav(audio, desired_channels=2)
        right_channel = audio[:, 1]
        right_channel = right_channel[:16000]
        wav_data = right_channel.numpy()

        wav_data_list.append(wav_data)
        jpg_files.append(jpg_file)
        labels.append(label)

    dataset = tf.data.Dataset.from_tensor_slices((wav_data_list, jpg_files, labels))
    datasets.append(dataset)

dataset = datasets[0]
for ds in datasets[1:]:
    dataset = dataset.concatenate(ds)

dataset = dataset.shuffle(buffer_size=1000)

train_dataset = dataset.take(int(0.7 * len(dataset)))
val_dataset = dataset.skip(int(0.7 * len(dataset)))

batch_size = 4

def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (1080, 1080))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image


train_dataset = train_dataset.map(lambda wav, jpg, label: (wav, load_image(jpg), label))
val_dataset = val_dataset.map(lambda wav, jpg, label: (wav, load_image(jpg), label))

train_dataset = train_dataset.map(lambda wav, jpg, label: ((wav, jpg), label))
val_dataset = val_dataset.map(lambda wav, jpg, label: ((wav, jpg), label))

train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(1080, 1080, 3))

wav_input = Input(shape=(16000,), dtype=tf.float32)
jpg_input = Input(shape=(1080, 1080, 3))

wav_input_reshaped = tf.reshape(wav_input, (-1, 160, 100))

lstm_output = LSTM(128)(wav_input_reshaped)

resnet_output = resnet(jpg_input)
resnet_output = tf.keras.layers.GlobalAveragePooling2D()(resnet_output)

# 合并LSTM输出和ResNet50输出
merged = Concatenate()([lstm_output, resnet_output])

output = Dense(64, activation='relu')(merged)
output = Dense(1, activation='linear')(output)

model = Model(inputs=[wav_input, jpg_input], outputs=output)

model.compile(optimizer='adam', loss='mse')

model.fit(train_dataset, epochs=20, validation_data=val_dataset)

y_true = []
y_pred = []

for (wav, jpg), label in val_dataset:
    y_true.extend(label.numpy())
    pred = model.predict((wav, jpg))
    y_pred.extend(pred.flatten())

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.4f}")

mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.4f}")

rmse = math.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape:.4f}")

r2 = r2_score(y_true, y_pred)
print(f"R2: {r2:.4f}")
model.save("xigua.keras")
