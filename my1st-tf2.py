import tensorflow as tf
tf.keras.backend.set_floatx('float64')
print("程式開始")
'''
import sys
print(sys.path)
'''
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
'''
print(x_train.shape) #(60000, 28, 28)
print(y_train.shape) #(60000,)

print(x_test.shape) #(10000, 28, 28)
print(y_test.shape) #(10000,)

#print(type(x_train)) #<class 'numpy.ndarray'>
#print(x_train.dtype) #uint8
#print(x_train)
'''
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
predictions = model(x_train[:1]).numpy()
predictions # 10個float, 0-9的評分
#print (x_train[:1])
#print(predictions)

tf.nn.softmax(predictions).numpy() # 似乎沒有必要，已經是numpy array

#print(predictions.shape) #(1, 10)
#print(predictions.dtype) #float64

import sys
sys.exit("先study code到這裡")

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
probability_model(x_test[:5])
