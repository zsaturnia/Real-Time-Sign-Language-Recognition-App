import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
to_categorical = tf.keras.utils.to_categorical



print("Loading and preprocessing data")

train_df = pd.read_csv(r"C:\Users\aymen\projetspython\sign_mnist_train.csv")
test_df = pd.read_csv(r"C:\Users\aymen\projetspython\sign_mnist_test.csv")

y_train = train_df['label'].values
y_test = test_df['label'].values
X_train = train_df.drop('label', axis=1).values
X_test = test_df.drop('label', axis=1).values

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

num_classes = len(np.unique(y_train)) + 1
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

print("Data preprocessed successfully.")

print("Building CNN model")

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)), 
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\nStarting model training")
history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=64,
                    validation_data=(X_test, y_test),
                    verbose=1)

print("Model training completed")

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nModel saved successfully as sign_language_model.h5")