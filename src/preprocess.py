# src/preprocess.py


from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(path, img_size=(64, 64), batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        zoom_range=0.2,
        shear_range=0.1,
        horizontal_flip=True
    )

    train = datagen.flow_from_directory(
        path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val = datagen.flow_from_directory(
        path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train, val
