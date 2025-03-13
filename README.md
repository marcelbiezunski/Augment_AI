# Augment_AI

Project provides data augmentation app for machine learning and computer vision. Augmentation was performed for the Google Cats and Dogs dataset.

Project setup on Windows:

1. Install Anaconda from official website.
2. Enter Anaconda Navigator.
3. Go to "environments" and press "import".
4. Choose from project location "environment.yml"
5. Press green button with arrow to run environment and choose "Open Terminal"
6. Type: cd "path/to/project/directory"
7. Type: python app.py

App.py is a Python-based application that allows users to perform custom data augmentation according to their specific needs. The tool provides flexibility in applying various augmentation techniques, making it useful for image preprocessing, deep learning, and computer vision projects.

AugmentLib.py is a dedicated library that contains all image transformations available in the appliaction.

Clean.py removes corrupted files from the dataset.
ClearAugmen.py restores the dataset to its original state by deleting augmented images.

