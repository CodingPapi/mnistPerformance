import tensorflow as tf
import time

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_set_count = len(train_labels)
test_set_count = len(test_labels)

t0 = time.time()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])


model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

t1 = time.time()
total_time = t1 - t0
print('\n')
print(f'Training set contained {train_set_count} images')
print(f'Testing set contained {test_set_count} images')
print(f'Model achieved {test_acc:.2f} accuracy')
print(f'Training and testing took {total_time:.2f} seconds')
