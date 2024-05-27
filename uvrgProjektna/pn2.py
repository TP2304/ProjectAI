import cv2
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp


def quickhull(points):
    # Funkcija, ki določi, na kateri strani premice p1-p2 leži točka p
    def get_side(p1, p2, p):
        return (p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0])

    # Funkcija za izračun razdalje od točke p do premice p1-p2
    def distance(p1, p2, p):
        return abs((p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0]))

    # Rekurzivna funkcija, ki doda točke k ovojnici (hull_points)
    def add_hull_points(p1, p2, points, hull_points):
        if not points:  # Če ni več točk, se funkcija konča
            return
        # Najdemo najbolj oddaljeno točko od premice p1-p2
        farthest_point = max(points, key=lambda p: distance(p1, p2, p))
        hull_points.append(farthest_point)  # Dodamo najbolj oddaljeno točko k ovojnici
        points.remove(farthest_point)  # Odstranimo to točko iz seznama točk

        # Razdelimo točke v dve skupini glede na stran, na kateri ležijo glede na premico p1-farthest_point in farthest_point-p2
        left_set = [p for p in points if get_side(p1, farthest_point, p) > 0]
        right_set = [p for p in points if get_side(farthest_point, p2, p) > 0]

        # Rekurzivno dodamo točke za levo in desno stran
        add_hull_points(p1, farthest_point, left_set, hull_points)
        add_hull_points(farthest_point, p2, right_set, hull_points)

    # Če je manj kot 3 točke, ni mogoče tvoriti ovojnice, zato vrnemo vse točke
    if len(points) < 3:
        return points

    # Najdemo točko z najmanjšo in največjo x koordinato
    min_x_point = min(points, key=lambda p: p[0])
    max_x_point = max(points, key=lambda p: p[0])

    # Inicializiramo ovojnico z začetnima točkama
    hull_points = [min_x_point, max_x_point]
    # Razdelimo preostale točke v dve skupini glede na stran, na kateri ležijo glede na premico min_x_point-max_x_point
    left_set = [p for p in points if get_side(min_x_point, max_x_point, p) > 0]
    right_set = [p for p in points if get_side(min_x_point, max_x_point, p) < 0]

    # Rekurzivno dodamo točke za levo in desno stran
    add_hull_points(min_x_point, max_x_point, left_set, hull_points)
    add_hull_points(max_x_point, min_x_point, right_set, hull_points)

    return hull_points  # Vrnemo točke ovojnice


image_path = r'C:\Users\klinc\Desktop\uvrgProjektna\sliki\image_2.jpg'
image = cv2.imread(image_path)  # Preberemo sliko z OpenCV

# Preverimo, ali je bila slika uspešno naložena
if image is None:
    print(f"Error: Could not load image at {image_path}")  # Če slika ni bila naložena, izpišemo napako
else:
    # Inicializiramo MediaPipe Face Mesh detekcijo in zaznavanje natančnih točk obraza
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)#nastavimoFM

    # Detekcija obraznih landmarkov
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:  # Če so detektirani obrazni landmarki
        for face_landmarks in results.multi_face_landmarks:
            # Določimo podmnožico koordinate, ki jih želimo uporabiti
            selected_indices = [
                10, 338, 297, 332, 284, 251, 389, 454, 356, 454, 323, 361, 288, 397, 365,
                379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                162, 21, 54, 103, 67, 109
            ]
            landmarks = []  # Inicializiramo seznam landmarkov
            for idx in selected_indices:  # Za vsak izbran indeks
                lm = face_landmarks.landmark[idx]  # Dobimo landmark za dani indeks
                x, y = int(lm.x * image.shape[1]), int(
                    lm.y * image.shape[0])  # Preračunamo koordinate landmarkov na sliko
                landmarks.append((x, y))  # Dodamo koordinate landmarkov v seznam

                # Uporabimo QuickHull za izračun zunanje meje landmarkov
            if len(landmarks) > 0:  # Če imamo vsaj en landmark
                hull_points = quickhull(landmarks)  # Izračunamo konveksno ovojnico landmarkov

                # Poskrbimo, da so točke ovojnice urejene v krožnem vrstnem redu
                hull_points = sorted(hull_points, key=lambda p: (np.arctan2(p[1] - np.mean([y for _, y in landmarks]),
                                                                            p[0] - np.mean([x for x, _ in
                                                                                            landmarks]))))  # Uredimo točke ovojnice

                # Narišemo točke ovojnice
                for i in range(len(hull_points)):  # Za vsako točko v ovojnici
                    start_point = hull_points[i]  # Začetna točka
                    end_point = hull_points[(i + 1) % len(hull_points)]  # Končna točka
                    cv2.line(image, start_point, end_point, (255, 0, 0), 1)  # Narišemo črto med točkami

                # Narišemo landmarke
                for point in landmarks:  # Za vsak landmark
                    cv2.circle(image, point, 2, (0, 255, 0), -1)  # Narišemo kroge na landmarkih

    # Prikaz rezultatov
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()  # Prikaz slike brez osi

# r'C:/Users/klinc/Desktop/uvrgProjekt/sliki/image_2.jpg'