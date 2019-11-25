import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random

# Variable used to stablish the training desired accuracy.
DESIRED_ACCURACY = 0.998

# Callback function usded to check the current epoch accuracy and end the training process if the desired accuracy is achieved.
class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('accuracy') > DESIRED_ACCURACY):
			print("\nReached " + str(DESIRED_ACCURACY * 100) + "% accuracy so cancelling training!")
			self.model.stop_training = True


callbacks = myCallback()

# Dataset load from TensorFlow MNIST Dataset
mnist = tf.keras.datasets.mnist
(orig_training_images, training_labels) , (orig_test_images, test_labels) = mnist.load_data()

# Fromating of the obtained training and testing datasets for its further usage
training_images_shape = orig_training_images.shape
testing_images_shape = orig_test_images.shape
print("Training images dataset shape: " + str(training_images_shape))
print("Testing images dataset shape: " + str(testing_images_shape))

training_images = orig_training_images.reshape(training_images_shape[0], training_images_shape[1], training_images_shape[2], 1)
training_images = training_images / 255.0

test_images = orig_test_images.reshape(testing_images_shape[0], testing_images_shape[1], testing_images_shape[2], 1)
test_images = test_images / 255.0


# Display examples of the dataset images
image_size = training_images_shape[1]
display_grid = np.zeros((image_size, image_size * 10))

for i in range(10):
	idx = np.where(test_labels == i)[0][0]
	orig_image = orig_test_images[idx]
	display_grid[:, i * image_size : (i + 1) * image_size] = orig_image

print("Press any key to continue...")
fig = plt.figure()
plt.axis('off')
plt.grid(False)
plt.imshow(display_grid, cmap='gray')
plt.draw()
plt.waitforbuttonpress(0)
plt.close(fig)


# Definition of the CNN model
model = tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (image_size, image_size, 1)),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation = 'relu'),
	tf.keras.layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics =['accuracy'])
model.summary()

# NOTICE: If you want to train the model comment the second block of code contained between the asterisks line [*] and leave the first one uncommented.
# However if you don't want to train the model and only desire to load the pretrained weights for the model comment the first block of code contained between the asteriks line [*] and leave the second one uncommented.

# Training of the CNN model with the selected training set, the obtained weights are stored in the given directory
# # ****************************************************
# model.fit(training_images, training_labels, epochs = 10, callbacks = [callbacks])
# model.save_weights('./checkpoints/numbers_model')
# ****************************************************

# Load the pŕetrained model weights from the given directory
# **************************************************
model.load_weights('./checkpoints/numbers_model')
# **************************************************

# Testing of the trained model with 10 randome testing image_size
for i in range(10):
	idx = random.randint(0, orig_test_images.shape[0])

	orig_image = orig_test_images[idx]
	image = orig_image.reshape(1, image_size, image_size, 1)
	image = image / 255.0
	image = tf.cast(image, tf.float32)

	prediction = list(list(model.predict(image))[0])

	print('\n')
	print("Press any key to continue...")
	print('Predicted: ' + str(prediction.index(max(prediction))))
	print('Real: ' + str(test_labels[idx]))

	fig = plt.figure()
	plt.axis('off')
	plt.grid(False)
	plt.imshow(orig_image, cmap='gray')
	plt.draw()
	plt.waitforbuttonpress(0)
	plt.close(fig)
