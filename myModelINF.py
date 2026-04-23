from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

model = load_model("image_classifier.keras")

img_path = "path for image (dog or cat)"
img = load_img(img_path, target_size = (128, 128))
img_array = img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]
label = "dog" if prediction > 0.5 else "cat"
conf = prediction if prediction > 0.5 else 1 - prediction
print(f"On the image: {label} with confidence: {conf*100:.1f}%")