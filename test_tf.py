import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.models import Model

# Define input layer for study data
study_input = Input(shape=(max_study_length,), dtype='int32', name='study_input')

# Define input layer for drug data
drug_input = Input(shape=(max_drug_length,), dtype='int32', name='drug_input')

# Define embedding layer for study data
study_embedding = Embedding(input_dim=num_study_words, output_dim=embedding_dim, input_length=max_study_length)(study_input)
study_embedding = Dropout(dropout_rate)(study_embedding)

# Define embedding layer for drug data
drug_embedding = Embedding(input_dim=num_drug_words, output_dim=embedding_dim, input_length=max_drug_length)(drug_input)
drug_embedding = Dropout(dropout_rate)(drug_embedding)

# Define LSTM layer for study data
study_lstm = LSTM(lstm_units)(study_embedding)

# Define LSTM layer for drug data
drug_lstm = LSTM(lstm_units)(drug_embedding)

# Concatenate the outputs of the LSTM layers
concatenated = tf.keras.layers.concatenate([study_lstm, drug_lstm])

# Add a fully connected layer
dense1 = Dense(dense_units, activation='relu')(concatenated)

# Add a dropout layer
dropout = Dropout(dropout_rate)(dense1)

# Add another fully connected layer
dense2 = Dense(dense_units, activation='relu')(dropout)

# Add an output layer
output = Dense(num_classes, activation='softmax')(dense2)

# Define the model
model = Model(inputs=[study_input, drug_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X_train_study, X_train_drug], y_train, epochs=num_epochs, batch_size=batch_size, validation_data=([X_val_study, X_val_drug], y_val))

# Evaluate the model
loss, accuracy = model.evaluate([X_test_study, X_test_drug], y_test, batch_size=batch_size)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Train the model
history = model.fit([X_train_study, X_train_drug], y_train, epochs=num_epochs, batch_size=batch_size, validation_data=([X_val_study, X_val_drug], y_val))

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Evaluate the model on the test set
y_pred = model.predict([X_test_study, X_test_drug])
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

# Calculate the F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print('Test F1 score:', f1)

from sklearn.metrics import roc_curve, auc
from itertools import cycle

# Predict the probabilities for each class for the test set
y_prob = model.predict([X_test_study, X_test_drug])

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test.shape[1]
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First, aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at these points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally, average and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot ROC curves for each class and micro/macro-average
plt.figure(figsize=(8, 6))
lw = 2
colors = cycle(['blue', 'red', 'green', 'yellow', 'purple', 'orange'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', lw=lw,
         label='Micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))

plt.plot(fpr["macro"], tpr["macro"], color='navy', lw=lw, linestyle='--',
         label='Macro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["macro"]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import confusion_matrix

# Predict the classes for the test set
y_pred = model.predict([X_test_study, X_test_drug])
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Plot confusion matrix
plt.imshow(conf_matrix, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(np.arange(n_classes), drug_encoder.inverse_transform(np.arange(n_classes)), rotation=45)
plt.yticks(np.arange(n_classes), study_encoder.inverse_transform(np.arange(n_classes)))
plt.colorbar()

for i in range(n_classes):
    for j in range(n_classes):
        plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')

plt.show()


