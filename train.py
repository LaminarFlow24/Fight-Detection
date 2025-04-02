import os
import shutil
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
import keras
from collections import deque
import matplotlib.pyplot as plt
plt.style.use("seaborn")


from IPython.display import HTML
from base64 import b64encode

from sklearn.model_selection import train_test_split

from keras.layers import Input, Dropout, Flatten, LSTM, Dense, TimeDistributed, Bidirectional
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical, plot_model
from keras.callbacks import EarlyStopping

def render_video(filepath):
    html_str = ''
    video_data = open(filepath,'rb').read()
    src = 'data:video/mp4;base64,' + b64encode(video_data).decode()
    html_str += '<video width=640 muted controls autoplay loop><source src="%s" type="video/mp4"></video>' % src
    return HTML(html_str)

def display_video(filepath):
    html_out = ''
    with open(filepath, 'rb') as f:
        data = f.read()
    src = 'data:video/mp4;base64,' + b64encode(data).decode()
    html_out += f'<video width=640 muted controls autoplay loop><source src="{src}" type="video/mp4"></video>'
    return HTML(html_out)

nonViolenceDir = "/kaggle/input/real-life-violence-situations-dataset/Real Life Violence Dataset/NonViolence/"
violenceDir = "/kaggle/input/real-life-violence-situations-dataset/Real Life Violence Dataset/Violence/"

nonViolenceFiles = os.listdir(nonViolenceDir)
violenceFiles = os.listdir(violenceDir)

randNonViolence = random.choice(nonViolenceFiles)
randViolence = random.choice(violenceFiles)

render_video(f"{nonViolenceDir}/{randNonViolence}")
render_video(f"{violenceDir}/{randViolence}")

IMG_HEIGHT, IMG_WIDTH = 64, 64
SEQ_LENGTH = 16
DATA_PATH = "../input/real-life-violence-situations-dataset/Real Life Violence Dataset/"
CLASS_NAMES = ["NonViolence", "Violence"]

def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(int(total / SEQ_LENGTH), 1)
    for i in range(SEQ_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
        norm = resized / 255.0
        frames.append(norm)
    cap.release()
    return frames

def build_dataset():
    feats = []
    labs = []
    paths = []
    for idx, name in enumerate(CLASS_NAMES):
        print(f'Processing class: {name}')
        file_list = os.listdir(os.path.join(DATA_PATH, name))
        for file in file_list:
            vid_path = os.path.join(DATA_PATH, name, file)
            frs = extract_frames(vid_path)
            if len(frs) == SEQ_LENGTH:
                feats.append(frs)
                labs.append(idx)
                paths.append(vid_path)
    feats = np.asarray(feats)
    labs = np.array(labs)
    return feats, labs, paths

features, labels, video_paths = build_dataset()

np.save("features.npy", features)
np.save("labels.npy", labels)
np.save("video_files_paths.npy", video_paths)

features, labels, video_paths = np.load("features.npy"), np.load("labels.npy"), np.load("video_files_paths.npy")

onehot = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(features, onehot, test_size=0.1, shuffle=True, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

from keras.applications.mobilenet_v2 import MobileNetV2
mobile_net = MobileNetV2(include_top=False, weights="imagenet")
mobile_net.trainable = True
for layer in mobile_net.layers[:-40]:
    layer.trainable = False

def create_model():
    model = Sequential()
    model.add(Input(shape=(SEQ_LENGTH, IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(TimeDistributed(mobile_net))
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Flatten()))
    lstm_fw = LSTM(32)
    lstm_bw = LSTM(32, go_backwards=True)
    model.add(Bidirectional(lstm_fw, backward_layer=lstm_bw))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(len(CLASS_NAMES), activation='softmax'))
    model.summary()
    return model

mobLSTM = create_model()
plot_model(mobLSTM, to_file='MobBiLSTM_model_structure_plot.png', show_shapes=True, show_layer_names=True)

early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=5, min_lr=0.00005, verbose=1)

mobLSTM.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])

history = mobLSTM.fit(x=X_train, y=y_train, epochs=50, batch_size=8, shuffle=True, validation_split=0.2, callbacks=[early_stop, reduce_lr])

eval_hist = mobLSTM.evaluate(X_test, y_test)

def plot_metric(history_obj, m1, m2, title_text):
    val1 = history_obj.history[m1]
    val2 = history_obj.history[m2]
    epochs = range(len(val1))
    plt.plot(epochs, val1, 'blue', label=m1)
    plt.plot(epochs, val2, 'orange', label=m2)
    plt.title(str(title_text))
    plt.legend()

plot_metric(history, 'loss', 'val_loss', 'Loss vs Validation Loss')
plot_metric(history, 'accuracy', 'val_accuracy', 'Accuracy vs Validation Accuracy')

preds = mobLSTM.predict(X_test)
pred_labels = np.argmax(preds, axis=1)
true_labels = np.argmax(y_test, axis=1)
print(true_labels.shape, pred_labels.shape)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred_labels, true_labels)
print('Accuracy Score is : ', acc)

import seaborn as sns
from sklearn.metrics import confusion_matrix
ax = plt.subplot()
cmat = confusion_matrix(true_labels, pred_labels)
sns.heatmap(cmat, annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['True', 'False'])
ax.yaxis.set_ticklabels(['NonViolence', 'Violence'])

from sklearn.metrics import classification_report
report = classification_report(true_labels, pred_labels)
print('Classification Report is : \n', report)

def predict_framewise(video_path, out_path, seq_len):
    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('m','p','4','v'),
                             cap.get(cv2.CAP_PROP_FPS), (orig_w, orig_h))
    q = deque(maxlen=seq_len)
    curr_pred = ''
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
        norm = resized / 255.0
        q.append(norm)
        if len(q) == seq_len:
            prob = mobLSTM.predict(np.expand_dims(q, axis=0))[0]
            idx = np.argmax(prob)
            curr_pred = CLASS_NAMES[idx]
        if curr_pred == "Violence":
            cv2.putText(frame, curr_pred, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 12)
        else:
            cv2.putText(frame, curr_pred, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 12)
        writer.write(frame)
    cap.release()
    writer.release()

plt.style.use("default")
def display_random_frames(videopath):
    plt.figure(figsize=(20,15))
    cap = cv2.VideoCapture(videopath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sorted(random.sample(range(SEQ_LENGTH, total_frames), 12))
    for count, idx in enumerate(indices, 1):
        plt.subplot(5,4,count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frm = cap.read()
        if not ret:
            break
        frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        plt.imshow(frm)
        plt.tight_layout()
    cap.release()

test_dir = 'test_videos'
os.makedirs(test_dir, exist_ok=True)
output_video = f'{test_dir}/Output-Test-Video.mp4'

input_video = "../input/real-life-violence-situations-dataset/Real Life Violence Dataset/Violence/V_378.mp4"
predict_framewise(input_video, output_video, SEQ_LENGTH)
display_random_frames(output_video)
render_video(input_video)

input_video = "../input/real-life-violence-situations-dataset/Real Life Violence Dataset/NonViolence/NV_1.mp4"
predict_framewise(input_video, output_video, SEQ_LENGTH)
display_random_frames(output_video)
render_video(input_video)

def predict_video_clip(video_path, seq_len):
    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    clip = []
    curr_pred = ''
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(int(total/seq_len), 1)
    for i in range(seq_len):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
        norm = resized / 255.0
        clip.append(norm)
    prob = mobLSTM.predict(np.expand_dims(clip, axis=0))[0]
    idx = np.argmax(prob)
    curr_pred = CLASS_NAMES[idx]
    print(f'Predicted: {curr_pred}\nConfidence: {prob[idx]}')
    cap.release()

input_video = "../input/real-life-violence-situations-dataset/Real Life Violence Dataset/Violence/V_276.mp4"
predict_video_clip(input_video, SEQ_LENGTH)
render_video(input_video)

input_video = "../input/real-life-violence-situations-dataset/Real Life Violence Dataset/NonViolence/NV_23.mp4"
predict_video_clip(input_video, SEQ_LENGTH)
render_video(input_video)

mobLSTM.save("mobLSTM_model_v2.h5")
mobLSTM.save("violence_detection_model.h5")
model_output = "/kaggle/working/violence_detection_model.h5"
shutil.move("violence_detection_model.h5", model_output)
shutil.move("mobLSTM_model_v2.h5", "/kaggle/working/mobLSTM_model_v2.h5")
