import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
import numpy as np

# Simulated data config
IMG_SIZE = 128
NUM_CLASSES_DISEASE = 4 
NUM_CLASSES_NPK = 3       
NUM_CLASSES_ROOT = 2      

# Simulate data (100 samples)
X = np.random.rand(100, IMG_SIZE, IMG_SIZE, 3)
y_disease = tf.keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES_DISEASE, 100), NUM_CLASSES_DISEASE)
y_npk = tf.keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES_NPK, 100), NUM_CLASSES_NPK)
y_root = tf.keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES_ROOT, 100), NUM_CLASSES_ROOT)

# Model definition
input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = Conv2D(32, (3,3), activation='relu')(input_layer)
x = MaxPooling2D()(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dropout(0.3)(x)

# Output heads
out_disease = Dense(NUM_CLASSES_DISEASE, activation='softmax', name='disease')(x)
out_npk = Dense(NUM_CLASSES_NPK, activation='softmax', name='npk')(x)
out_root = Dense(NUM_CLASSES_ROOT, activation='softmax', name='root')(x)

# Final model
model = Model(inputs=input_layer, outputs=[out_disease, out_npk, out_root])
model.compile(
    optimizer='adam',
    loss={
        'disease': 'categorical_crossentropy',
        'npk': 'categorical_crossentropy',
        'root': 'categorical_crossentropy'
    },
    metrics={
        'disease': 'accuracy',
        'npk': 'accuracy',
        'root': 'accuracy'
    }
)


# Train model
model.fit(X, {'disease': y_disease, 'npk': y_npk, 'root': y_root}, epochs=5, batch_size=8)

# Save model
model.save('plant_disease_model.h5')
print("✅ Model saved as plant_disease_model.h5")
