import cv2
import random
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from tqdm import tqdm
import random

def get_image_names(folder):
    image_names = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_names.append(filename)
    return image_names


def random_crop(img_path, min_size=(100, 100), max_size=(500, 500)):
  img = cv2.imread(img_path)
  h, w = img.shape[:2]

  crop_height = random.randint(min_size[0], min(h, max_size[0]))
  crop_width = random.randint(min_size[1], min(w, max_size[1]))

  y = random.randint(0, h - crop_height)
  x = random.randint(0, w - crop_width)

  return img[y:y+crop_height, x:x+crop_width]

def random_rotate(image_path, angle_range=(0, 360)):
  img = cv2.imread(image_path)

  angle = random.randint(angle_range[0], angle_range[1])

  (h, w) = img.shape[:2]

  center = (w // 2, h // 2)

  M = cv2.getRotationMatrix2D(center, angle, 0.5)

  rotated_img = cv2.warpAffine(img, M, (w, h))

  return rotated_img

def flip_image(image_path):
  img = cv2.imread(image_path)

  rotation_type = random.choice(["vertical", "horizontal"])   

  if rotation_type == "vertical":
    rotated = cv2.flip(img, 0)

  else:
    rotated = cv2.flip(img, 1)
  return rotated

def random_resize(image_path):
    img = cv2.imread(image_path)

    scale_x = np.random.uniform(0.5, 2.0)
    scale_y = np.random.uniform(0.5, 2.0)

    new_width = int(img.shape[1] * scale_x)
    new_height = int(img.shape[0] * scale_y)

    resized_img = cv2.resize(img, (new_width, new_height))

    return resized_img

def random_brightness(img_path, factor = random.uniform(0.5,1.2)):
  img = cv2.imread(img_path)

  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  hsv[:,:,2] = hsv[:,:,2] * factor

  hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)

  new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

  return new_img

def adjust_contrast(img_path, contrast_factor = random.uniform(0.25,1.75)):
    image = cv2.imread(img_path)

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(clipLimit=contrast_factor, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    adjusted_lab_image = cv2.merge([l_channel, a_channel, b_channel])

    adjusted_image = cv2.cvtColor(adjusted_lab_image, cv2.COLOR_LAB2BGR)

    return adjusted_image

def add_gaussian_noise(image_path, mean=random.uniform(0,10), std=random.uniform(5,25)):
    image = cv2.imread(image_path)

    noise = np.random.normal(mean, std, image.shape)

    img_float64 = image.astype(np.float64)

    img_noise = img_float64 + noise

    img_noise = np.clip(img_noise, 0, 255)

    img_noise = img_noise.astype(np.uint8)

    return img_noise

def add_salt_and_pepper_noise_color(image_path, prob=0.25):
    image = cv2.imread(image_path)
    output = np.copy(image)
    
    mask = np.random.choice([0, 1, 2], size=image.shape, p=[1-prob, prob/2, prob/2])

    output[mask == 1] = 0
    output[mask == 2] = 255
    return output

def random_perspective_points(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    min_x, max_x = 0, w-1
    min_y, max_y = 0, h-1

    points = np.random.randint(min_x, max_x, size=(4, 2))

    points = points[np.argsort(points[:, 1])]
    points[:2] = points[:2][np.argsort(points[:2, 0])]
    points[2:] = points[2:][np.argsort(points[2:, 0])[::-1]] 

    return points

def perspective_transform(img_path, points):
    img = cv2.imread(img_path)

    h, w = img.shape[:2]

    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (w, h))

    return dst

def autoencoder(img_path, num_images):
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
      try:
          tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
          logical_gpus = tf.config.experimental.list_logical_devices('GPU')
          print(f"Number of Logical GPUs: {len(logical_gpus)}")
      except RuntimeError as e:
          print(e)
  else:
      print("No GPUs available. Training on CPU.")

  data_dir = img_path

  train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

  train_generator = train_datagen.flow_from_directory(
      data_dir,
      target_size=(128, 128),
      batch_size=64,
      class_mode='input',
      subset='training',
      shuffle=True)

  val_generator = train_datagen.flow_from_directory(
      data_dir,
      target_size=(128, 128),
      batch_size=64,
      class_mode='input',
      subset='validation',
      shuffle=False)

  def build_autoencoder(input_shape):
      encoder_input = layers.Input(shape=input_shape)
      x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
      x = layers.MaxPooling2D((2, 2), padding='same')(x)
      x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
      x = layers.MaxPooling2D((2, 2), padding='same')(x)
      encoded = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

      x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
      x = layers.UpSampling2D((2, 2))(x)
      x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
      x = layers.UpSampling2D((2, 2))(x)
      decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

      autoencoder = models.Model(encoder_input, decoded)
      return autoencoder

  autoencoder = build_autoencoder((128, 128, 3))
  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

  autoencoder.fit(
      train_generator,
      epochs=50,
      validation_data=val_generator,
      steps_per_epoch=len(train_generator),
      validation_steps=len(val_generator)
  )

  output_dir = data_dir

  processed_images = 0

  for i in tqdm(range(len(train_generator))):
      batch = train_generator[i][0]
      batch_size = batch.shape[0]

      if num_images - processed_images < batch_size:
          indices_to_process = random.sample(range(batch_size), num_images - processed_images)
      else:
          indices_to_process = range(batch_size)
      
      print(f"Processing indices in batch {i}: {indices_to_process}")

      for local_index in indices_to_process:
          img = autoencoder.predict(batch[local_index:local_index+1])
          img = np.squeeze(img)
          img = (img * 255).astype(np.uint8)
          img = array_to_img(img)
          img.save(os.path.join(output_dir, f"generated_img_{i}_{local_index}.jpg"))
          processed_images += 1

          if processed_images >= num_images:
              break

      if processed_images >= num_images:
          break
