import tensorflow as tf

filename = 'data/201807-202003/saved_model_50_0.6218512654304504'
new_model = tf.keras.models.load_model(filename)

# Check its architecture
print(new_model.summary())
