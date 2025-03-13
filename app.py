from PIL import Image
import os
import cv2
import sys
import random

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
    perspective_transform,
    autoencoder    
)


def get_image_names(folder):
    image_names = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_names.append(filename)
    return image_names


def main():
    image_names = get_image_names("photos")

    print("percentage of images to augment: ")
    print("[0] 25%")
    print("[1] 50%")
    print("[2] 75%")
    print("[3] 100%")
    print("[4] exit application")

    percentage = int(input("enter here: "))
    match percentage:
        case 0:
            num_images = int(len(image_names) * 0.25)
        case 1:
            num_images = int(len(image_names) * 0.5)
        case 2:
            num_images = int(len(image_names) * 0.75)
        case 3:
            num_images = len(image_names)
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
    print("[9] autoencoder ")
    print("[10] exit application")

    choice = int(input("enter here: "))

    match choice:
        case 0:
            print("WARNING: This method is not recommended for medical images, where every pixel is important")
            num_images_random = random.sample(image_names,num_images)
            for image_path in num_images_random:
                new_image = random_crop("photos/" + image_path)
                cv2.imwrite(f"photos/{image_path[:-4]}_cropped.jpg", new_image)
        
        case 1:
            print("WARNING: This method is not recommended for images with clear orientation, e.g. text image")
            num_images_random = random.sample(image_names,num_images)
            for image_path in num_images_random:
                new_image = random_rotate("photos/" + image_path)
                cv2.imwrite(f"photos/{image_path[:-4]}_rotated.jpg", new_image)

        case 2:
            print("WARNING: This method is not recommended for images with clear orientation, e.g. car image")
            num_images_random = random.sample(image_names,num_images)
            for image_path in num_images_random:
                new_image = flip_image("photos/" + image_path)
                cv2.imwrite(f"photos/{image_path[:-4]}_flipped.jpg", new_image)
        
        case 3:
            print("WARNING: This method is not recommended for images where size matter, e.g. medical image")
            num_images_random = random.sample(image_names,num_images)
            for image_path in num_images_random:
                new_image = random_resize("photos/" + image_path)
                cv2.imwrite(f"photos/{image_path[:-4]}_resised.jpg", new_image)

        case 4:
            print("WARNING: This method is not recommended for images where contrast matter, e.g. X-ray image")
            num_images_random = random.sample(image_names,num_images)
            for image_path in num_images_random:
                new_image = random_brightness("photos/" + image_path)
                cv2.imwrite(f"photos/{image_path[:-4]}_brightened.jpg", new_image)

        case 5:
            print("WARNING: This method is not recommended for images where contrast matter, e.g. X-ray image")
            num_images_random = random.sample(image_names,num_images)
            for image_path in num_images_random:
                new_image = adjust_contrast("photos/" + image_path)
                cv2.imwrite(f"photos/{image_path[:-4]}_adjust_contrast.jpg", new_image)

        case 6:
            print("WARNING: This method is not recommended for images where noise may lead to misdiagnosis, e.g. medical image")
            num_images_random = random.sample(image_names,num_images)
            for image_path in num_images_random:
                new_image = add_gaussian_noise("photos/" + image_path)
                cv2.imwrite(f"photos/{image_path[:-4]}_gaussian.jpg", new_image)

        case 7:
            print("WARNING: This method is not recommended for images where noise may lead to misdiagnosis, e.g. medical image")
            num_images_random = random.sample(image_names,num_images)
            for image_path in num_images_random:
                new_image = add_salt_and_pepper_noise_color("photos/" + image_path)
                cv2.imwrite(f"photos/{image_path[:-4]}_salt_and_pepper.jpg", new_image)

        case 8:
            print("WARNING: This method is not recommended for images where perspective is key, e.g. architecture image")
            num_images_random = random.sample(image_names,num_images)
            for image_path in num_images_random:
                points = random_perspective_points("photos/" + image_path)
                new_image = perspective_transform("photos/" + image_path, points)
                cv2.imwrite(f"photos/{image_path[:-4]}_perspective_transform.jpg", new_image)

        case 9:
            try:
                print("WARNING: This method is time consuming, be patient")
                img_path = "photos/"
                autoencoder(img_path,num_images)
            except ValueError as e:
                print(f"Error: {e}. Please make sure that the images are placed in a separate folder and that the folder is copied to the 'photos' directory.")
            except Exception as e:
                print(f"Unexpected error: {e}")
        
        case 10:
            sys.exit("Have a nice day!")

    
if __name__ == "__main__":
    while True:
        main()
