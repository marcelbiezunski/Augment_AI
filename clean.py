from PIL import Image, ImageFile
import os


ImageFile.LOAD_TRUNCATED_IMAGES = True

def remove_corrupted_images(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                with Image.open(filepath) as img:
                    img.verify()
            except (IOError, SyntaxError, Image.UnidentifiedImageError):
                print(f"Usuwam uszkodzony plik: {filepath}")
                os.remove(filepath)

# example:
remove_corrupted_images('petImages/train')
remove_corrupted_images('petImages/validation')
