# main.py

import tensorflow as tf
from src.preprocess import get_data_generators
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout, Input

def build_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        Input(shape=input_shape),  # ✅ Recommended way to set input
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        Dropout(0.5),  # ✅ Helps reduce overfitting
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    # ✅ Use smaller image size for faster training
    train_gen, val_gen = get_data_generators("data/PlantVillage", img_size=(64, 64), batch_size=32)

    model = build_model((64, 64, 3), train_gen.num_classes)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1)
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=3,  # 🔁 Quick test; increase later
        callbacks=callbacks
    )

    model.save("models/plant_disease_model.h5")
    print("✅ Model saved.")

if __name__ == '__main__':
    train_model()
