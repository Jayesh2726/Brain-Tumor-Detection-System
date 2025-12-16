import tensorflow as tf

# Load original model
model = tf.keras.models.load_model('BrainTumor10Epochs.h5')

# Save fixed version
model.save('BrainTumor10Epochs_fixed.h5', include_optimizer=False)

print("✅ Model converted successfully!")