from PIL import Image, ImageFile
import os

# Włączamy obsługę uszkodzonych obrazów
ImageFile.LOAD_TRUNCATED_IMAGES = True

def remove_corrupted_images(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                with Image.open(filepath) as img:
                    img.verify()  # Sprawdzanie integralności obrazu
            except (IOError, SyntaxError, Image.UnidentifiedImageError):
                print(f"Usuwam uszkodzony plik: {filepath}")
                os.remove(filepath)

# Przykład użycia
remove_corrupted_images('petImages/train')
remove_corrupted_images('petImages/validation')