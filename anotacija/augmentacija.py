import os
import cv2
import albumentations as alb
import time
from datetime import datetime

def ustvari_transformacije():
    return alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.RandomBrightnessContrast(p=0.3),
        alb.OneOf([
            alb.MotionBlur(blur_limit=7),
            alb.GaussianBlur(blur_limit=7),
            alb.GaussNoise(var_limit=(10.0, 50.0))
        ], p=0.3),
        alb.Rotate(limit=20, p=0.5),
        alb.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
        alb.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        alb.RandomGamma(gamma_limit=(80, 120), p=0.5),
        alb.ToGray(p=0.2),
    ])

if __name__ == '__main__':
    izhodna_mapa = r'D:\Users\Stefi\Desktop\projektniPraktikum\augmented_images'
    vhodna_mapa = r'D:\Users\Stefi\Desktop\projektniPraktikum\slike'
    dat = 'slika1.png'
    pot_slike = os.path.join(vhodna_mapa, dat)
    try:
        os.makedirs(izhodna_mapa, exist_ok=True)
        slika = cv2.imread(pot_slike)

        if slika is not None:
            slika = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)
            original_height, original_width = slika.shape[:2]
            veriga_transformacij = ustvari_transformacije()
            zacetek = time.time()
            i = 0
            while i < 100:
                povecana_slika = veriga_transformacij(image=slika)
                koncna_slika = povecana_slika['image']
                koncna_slika = cv2.resize(koncna_slika, (original_width, original_height))
                dml = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  
                izhodna_pot = os.path.join(izhodna_mapa, f'augmented_image_{i}_{dml}.png')
                cv2.imwrite(izhodna_pot, cv2.cvtColor(koncna_slika, cv2.COLOR_RGB2BGR))
                print(f"Shranjena slika {i}: {izhodna_pot}")
                i += 1
            konec = time.time() - zacetek
            print(f"Vse slike so bile obdelane v {konec:.2f} sekundah")
        else:
            raise ValueError("Slika ni najdena ali ni pravilno nalozena")
    except Exception as e:
        print(f"Prislo je do napake: {str(e)}!!!")
