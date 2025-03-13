import cv2
import random
import numpy as np
from PIL import Image, ImageEnhance
import os

def random_crop(img_path, min_size=(30, 30), max_size=(500, 500)):
  image = cv2.imread(img_path)
  if image is None:
    # Handle loading error (e.g., print message, skip image)
    print(f"Error loading image: {img_path}")
    os.remove(img_path)
    print(f"Usunięto plik: {img_path}")
    return None  # Or raise an exception
  # Load the image
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
  # Wczytaj obraz przy użyciu OpenCV
    image = cv2.imread(img_path)

    # Konwersja obrazu do przestrzeni LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Podział na kanały L, A, B
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Zastosowanie CLAHE na kanale L z regulowanym clipLimit
    clahe = cv2.createCLAHE(clipLimit=contrast_factor, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    # Ponowne łączenie kanałów
    adjusted_lab_image = cv2.merge([l_channel, a_channel, b_channel])

    # Konwersja obrazu z powrotem do przestrzeni barwnej BGR
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
    if image is None:
      # Handle loading error (e.g., print message, skip image)
      print(f"Error loading image: {image_path}")
      return None  # Or raise an exception
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
