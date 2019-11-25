import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Size of the images to be processed [28 x 28]
image_size = 28

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

# Load the pŕetrained model weights from the given directory
model.load_weights('./checkpoints/numbers_model')


# Read all the stored images in the img directory
files = []

for r,d,f in os.walk('./img/'):
	for file in f:
		if '.jpg' in file:
			files.append(os.path.join(r, file))

# Predict the stored images types using the pretrained DNN model
for f in files:
	orig_image = Image.open(f)
	orig_image = np.array(orig_image)[:,:,0]

	image = orig_image.reshape(1, image_size, image_size, 1)
	image = image / 255.0
	image = tf.cast(image, tf.float32)

	prediction = list(list(model.predict(image))[0])

	print('\n')
	print("Press any key to continue...")
	print('Predicted: ' + str(prediction.index(max(prediction))))

	fig = plt.figure()
	plt.axis('off')
	plt.grid(False)
	plt.imshow(orig_image 	, cmap='gray')
	plt.draw()
	plt.waitforbuttonpress(0)
	plt.close(fig)
