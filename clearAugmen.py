import os

def delete_data(catalog):
    for file in os.listdir(catalog):
        if "_" in file:
            file_directory = os.path.join(catalog, file)
            try:
                os.remove(file_directory)
                print(f"File deleted: {file_directory}")
            except OSError as e:
                print(f"File removal failed: {file_directory}: {e}")

if __name__ == "__main__":
    catalog = "PetImages/train/Cat"
    delete_data(catalog)
    catalog = "PetImages/train/Dog"
    delete_data(catalog)

# catalog = location where augmented files are stored
