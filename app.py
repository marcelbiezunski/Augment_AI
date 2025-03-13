from PIL import Image
import os
import cv2
import sys
import random


def remove_unreadable_image(img_path):
  """Removes an image file if it cannot be read by OpenCV.

  Args:
    img_path: Path to the image file.
  """

  image = cv2.imread(img_path)
  if image is None:
    # Handle loading error (e.g., print message, skip image)
    print(f"Error loading image: {img_path}")
    os.remove(img_path)
    print(f"Usunięto plik: {img_path}")

from augmenLib import (
    random_crop,
    random_rotate,
    flip_image,
    random_resize,
    random_brightness,
    adjust_contrast,
    add_gaussian_noise,
    add_salt_and_pepper_noise_color,
    random_perspective_points,
    perspective_transform    
)


def get_image_names(folder):
    image_names = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_names.append(filename)
    return image_names


def main():
    cats_names = get_image_names("PetImages/train/Cat")
    dogs_names = get_image_names("PetImages/train/Dog")

    print("percentage of images to augment: ")
    print("[0] 25%")
    print("[1] 50%")
    print("[2] 75%")
    print("[3] 100%")
    print("[4] exit application")

    percentage = int(input("enter here: "))
    match percentage:
        case 0:
            num_images_cats = int(len(cats_names) * 0.25)
            num_images_dogs = int(len(dogs_names) * 0.25)
        case 1:
            num_images_cats = int(len(cats_names) * 0.5)
            num_images_dogs = int(len(dogs_names) * 0.5)
        case 2:
            num_images_cats = int(len(cats_names) * 0.75)
            num_images_dogs = int(len(dogs_names) * 0.75)
        case 3:
            num_images_cats = len(cats_names)
            num_images_dogs = len(dogs_names)
        case 4:
            sys.exit("Have a nice day!")


    print("Choose an augmentation method: ")
    print("[0] random_crop")
    print("[1] random_rotate")
    print("[2] flip_image")
    print("[3] random_resize ")
    print("[4] random_brightness ")
    print("[5] adjust_contrast ")
    print("[6] add_gaussian_noise ")
    print("[7] add_salt_and_pepper_noise_color ")
    print("[8] perspective_transform ")
    print("[9] exit application")

    choice = int(input("enter here: "))

    match choice:
        case 0:
            cats_names_random = random.sample(cats_names, num_images_cats)
            for image_path in cats_names_random:
                #remove_unreadable_image("PetImages/train/Cat/" + image_path)
                new_image = random_crop("PetImages/train/Cat/" + image_path)
                cv2.imwrite(f"PetImages/train/Cat/{image_path}_cropped.jpg", new_image)
            
            dogs_names_random = random.sample(dogs_names, num_images_dogs)
            for image_path in dogs_names_random:
                #remove_unreadable_image("PetImages/train/Dog/" + image_path)
                new_image = random_crop("PetImages/train/Dog/" + image_path)
                cv2.imwrite(f"PetImages/train/Dog/{image_path}_cropped.jpg", new_image)
        
        case 1:
            cats_names_random = random.sample(cats_names, num_images_cats)
            for image_path in cats_names_random:
                new_image = random_rotate("PetImages/train/Cat/" + image_path)
                cv2.imwrite(f"PetImages/train/Cat/{image_path}_rotated.jpg", new_image)
            
            dogs_names_random = random.sample(dogs_names, num_images_dogs)
            for image_path in dogs_names_random:
                new_image = random_rotate("PetImages/train/Dog/" + image_path)
                cv2.imwrite(f"PetImages/train/Dog/{image_path}_rotated.jpg", new_image)

        case 2:
            cats_names_random = random.sample(cats_names, num_images_cats)
            for image_path in cats_names_random:
                new_image = flip_image("PetImages/train/Cat/" + image_path)
                cv2.imwrite(f"PetImages/train/Cat/{image_path}_flipped.jpg", new_image)
            
            dogs_names_random = random.sample(dogs_names, num_images_dogs)
            for image_path in dogs_names_random:
                new_image = flip_image("PetImages/train/Dog/" + image_path)
                cv2.imwrite(f"PetImages/train/Dog/{image_path}_flipped.jpg", new_image)
        
        case 3:
            cats_names_random = random.sample(cats_names, num_images_cats)
            for image_path in cats_names_random:
                new_image = random_resize("PetImages/train/Cat/" + image_path)
                cv2.imwrite(f"PetImages/train/Cat/{image_path}_resised.jpg", new_image)
            
            dogs_names_random = random.sample(dogs_names, num_images_dogs)
            for image_path in dogs_names_random:
                new_image = random_resize("PetImages/train/Dog/" + image_path)
                cv2.imwrite(f"PetImages/train/Dog/{image_path}_resised.jpg", new_image)

        case 4:
            cats_names_random = random.sample(cats_names, num_images_cats)
            for image_path in cats_names_random:
                new_image = random_brightness("PetImages/train/Cat/" + image_path)
                cv2.imwrite(f"PetImages/train/Cat/{image_path}_brightened.jpg", new_image)
            
            dogs_names_random = random.sample(dogs_names, num_images_dogs)
            for image_path in dogs_names_random:
                new_image = random_brightness("PetImages/train/Dog/" + image_path)
                cv2.imwrite(f"PetImages/train/Dog/{image_path}_brightened.jpg", new_image)

        case 5:
            cats_names_random = random.sample(cats_names, num_images_cats)
            for image_path in cats_names_random:
                new_image = adjust_contrast("PetImages/train/Cat/" + image_path)
                cv2.imwrite(f"PetImages/train/Cat/{image_path}_adjust_contrast.jpg", new_image)
            
            dogs_names_random = random.sample(dogs_names, num_images_dogs)
            for image_path in dogs_names_random:
                new_image = adjust_contrast("PetImages/train/Dog/" + image_path)
                cv2.imwrite(f"PetImages/train/Dog/{image_path}_adjust_contrast.jpg", new_image)

        case 6:
            cats_names_random = random.sample(cats_names, num_images_cats)
            for image_path in cats_names_random:
                new_image = add_gaussian_noise("PetImages/train/Cat/" + image_path)
                cv2.imwrite(f"PetImages/train/Cat/{image_path}_gaussian.jpg", new_image)
            
            dogs_names_random = random.sample(dogs_names, num_images_dogs)
            for image_path in dogs_names_random:
                new_image = add_gaussian_noise("PetImages/train/Dog/" + image_path)
                cv2.imwrite(f"PetImages/train/Dog/{image_path}_gaussian.jpg", new_image)

        case 7:
            cats_names_random = random.sample(cats_names, num_images_cats)
            for image_path in cats_names_random:
                new_image = add_salt_and_pepper_noise_color("PetImages/train/Cat/" + image_path)
                cv2.imwrite(f"PetImages/train/Cat/{image_path}_salt_and_pepper.jpg", new_image)
            
            dogs_names_random = random.sample(dogs_names, num_images_dogs)
            for image_path in dogs_names_random:
                new_image = add_salt_and_pepper_noise_color("PetImages/train/Dog/" + image_path)
                cv2.imwrite(f"PetImages/train/Dog/{image_path}_salt_and_pepper.jpg", new_image)

        case 8:
            cats_names_random = random.sample(cats_names, num_images_cats)
            for image_path in cats_names_random:
                points = random_perspective_points("PetImages/train/Cat/" + image_path)
                new_image = perspective_transform("PetImages/train/Cat/" + image_path, points)
                cv2.imwrite(f"PetImages/train/Cat/{image_path}_perspective_transform.jpg", new_image)

            dogs_names_random = random.sample(dogs_names, num_images_dogs)
            for image_path in dogs_names_random:
                points = random_perspective_points("PetImages/train/Dog/" + image_path)
                new_image = perspective_transform("PetImages/train/Dog/" + image_path, points)
                cv2.imwrite(f"PetImages/train/Dog/{image_path}_perspective_transform.jpg", new_image)
        
        case 9:
            sys.exit("Have a nice day!")

    
if __name__ == "__main__":
    while True:
        main()
