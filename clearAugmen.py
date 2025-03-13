import os

def usun_zdjecia_z_podkresleniem(katalog):
    """Usuwa zdjęcia zawierające znak podkreślenia z danego katalogu.

    Args:
        katalog: Ścieżka do katalogu, w którym mają być usunięte zdjęcia.
    """

    for plik in os.listdir(katalog):
        if "_" in plik:  # Sprawdzenie, czy nazwa pliku zawiera znak podkreślenia
            sciezka_pliku = os.path.join(katalog, plik)
            try:
                os.remove(sciezka_pliku)
                print(f"Usunięto plik: {sciezka_pliku}")
            except OSError as e:
                print(f"Błąd podczas usuwania pliku {sciezka_pliku}: {e}")

if __name__ == "__main__":
    katalog = "PetImages/train/Cat"
    usun_zdjecia_z_podkresleniem(katalog)
    katalog = "PetImages/train/Dog"
    usun_zdjecia_z_podkresleniem(katalog)