# type: ignore
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, auc ,classification_report
import seaborn as sns
from sklearn.preprocessing import label_binarize
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob
from tensorflow.keras import optimizers
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
print(tf.__version__)

dataset_path = "/content/rice_dataset/rice_leaf_diseases"
class_names = os.listdir(dataset_path)
print("Classes found:", class_names)

image_counts, image_paths = get_image_stats(dataset_path, class_names)

# stats
print("Image counts per class:", image_counts)
plot_class_distribution(image_counts)

# display
show_sample_images(image_paths, class_names, num_images=3)

image_size = 224
batch_size = 20

# training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1/255)

dataset_train = dataset_path

train_dataset = train_datagen.flow_from_directory(
    dataset_train,
    shuffle=True,
    batch_size=batch_size,
    target_size=(image_size, image_size),
    subset='training',
    seed=66
)

validation_dataset = train_datagen.flow_from_directory(
    dataset_path,
    batch_size=batch_size,
    target_size=(image_size, image_size),
    subset='validation',
    seed=66,
    shuffle=False
)

print(train_dataset.class_indices)
print(validation_dataset.class_indices)
print(validation_dataset.classes)


# Sequential
image_size = 224
input_shape = (image_size,image_size,3)

model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape = input_shape, activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=input_shape, activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=input_shape, activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(3,activation='softmax'))

model.summary()


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, dpi=50)

# compile model before training 
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(learning_rate=2e-4),
    metrics=['acc']
)

# training
history = model.fit(train_dataset,
          epochs=50,
          validation_data=validation_dataset)

# save model
model.save('rice_model.keras')


# Plot the accuracy and loss curves
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# scores
scores = model.evaluate(validation_dataset, verbose=0)
print("test Accuracy: %.2f%%" % (scores[1]*100))

scores = model.evaluate(train_dataset, verbose=0)
print("train Accuracy: %.2f%%" % (scores[1]*100))

# confusion matriximport numpy as np
class_names = list(validation_dataset.class_indices.keys())

validation_dataset.reset()

predictions = model.predict(validation_dataset, verbose=1)

true_labels = validation_dataset.classes
predicted_labels = np.argmax(predictions, axis=1)

# calculate confusion
cm = confusion_matrix(true_labels, predicted_labels)

# heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_names, yticklabels=class_names)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# F1 Score
f1 = f1_score(true_labels, predicted_labels, average='weighted')

# precision
precision = precision_score(true_labels, predicted_labels, average='weighted')

# recall
recall = recall_score(true_labels, predicted_labels, average='weighted')

if len(np.unique(true_labels)) > 2:
    true_labels_one_hot = np.eye(len(np.unique(true_labels)))[true_labels]
    roc_auc = roc_auc_score(true_labels_one_hot, predictions, multi_class='ovr')
else:
    roc_auc = roc_auc_score(true_labels, predictions[:,1])

# Print scores
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")



true_labels_bin = label_binarize(true_labels, classes=list(range(len(class_names))))
n_classes = predictions.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot the ROC curves
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f"Class: {class_names[i]} (AUC = {roc_auc[i]:.2f})")

# Diagonal reference line
plt.plot([0, 1], [0, 1], 'k--', label="Chance")

# Plot settings
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Get true labels and predictions
y_true = validation_dataset.classes
y_pred = np.argmax(model.predict(validation_dataset), axis=1)

# Print report
print(classification_report(
    y_true,
    y_pred,
    target_names=list(train_dataset.class_indices.keys())
))

# model implmentation 

# load url
def load_image_from_url(url):
    req = urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1) # 'Load it as it is'
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB

# url
leaf = 'https://shorturl.at/iHDv0'
# leaf = 'https://shorturl.at/yRcpB'
# leaf = 'https://shorturl.at/kI4u7'


# image resize like original data
img = load_image_from_url(leaf)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

model = tf.keras.models.load_model('rice_model.keras')
pred = model.predict(img)
class_idx = np.argmax(pred[0])
confidence = np.max(pred[0])

# dispaly
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(load_image_from_url(leaf))
plt.title("Input Image")
plt.axis('off')

plt.subplot(1,2,2)
bars = plt.barh(class_names, pred[0], color=['green' if x==class_idx else 'gray' for x in range(len(class_names))])
plt.bar_label(bars, fmt='%.2f')
plt.title("Prediction Probabilities")
plt.tight_layout()
plt.show()

print(f"\n Final Prediction: {class_names[class_idx]} ({confidence:.1%} confidence)")