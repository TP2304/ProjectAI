# ProjectAI
Aplikacija za zaznavanje obraza in oči, ki uporablja Tkinter za uporabniški vmesnik in OpenCV za obdelavo slik.

## Specifikacije Projekta

### Namen Projekta
Aplikacija za zaznavanje obrazov in oči z uporabo umetne inteligence, ki omogoča identifikacijo znanih obrazov in spremljanje stanja oči.

### Predvideno Občinstvo in Uporaba
- **Kdo bo uporabljal**: Uporabniki, ki potrebujejo orodje za zaznavanje obrazov in spremljanje oči v realnem času.
- **Uporaba**: Lahko se uporablja v varnostnih sistemih, za spremljanje utrujenosti voznika ali za druge namene prepoznavanja obraza.

### Potrebe Uporabnikov
- **Kaj lahko pričakujejo**: 
  - Natančno zaznavanje obrazov in oči.
  - Prepoznavanje znanih obrazov.
  - Realnočasovno spremljanje stanja oči in predvajanje opozoril ob zaprtih očeh.

### Predpostavke in Odvisnosti
- **Kaj predvidevamo**: 
  - Uporabniki imajo osnovno znanje uporabe Python in Tkinter.
  - Aplikacija deluje na sistemih z nameščenim Python in potrebnimi knjižnicami (cv2, dlib, PIL, numpy).
- **Odvisnosti**:
  - Datoteke obrazov za prepoznavanje morajo biti na voljo.
  - Model `shape_predictor_68_face_landmarks.dat` mora biti prenesen in dostopen v projektu.

### Sistemske Funkcije in Zahteve
- **Zahteve**: 
  - Python 3.7 ali novejši.
  - Knjižnice: `opencv-python`, `face_recognition`, `dlib`, `numpy`, `Pillow`, `imutils`, `pygame`, `tkinter`.
- **Podrobnosti o funkcijah**:
  - **Zaznavanje obraza**: Uporablja Haarov kaskadni klasifikator in dlib za zaznavanje obrazov.
  - **Zaznavanje oči**: Uporablja Haarov kaskadni klasifikator za oči.
  - **Prepoznavanje obrazov**: Uporablja `face_recognition` za prepoznavanje obrazov iz vnaprej določenih slik.
  - **Sledenje stanju oči**: Spremlja razmerje oči za zaznavanje zaprtih oči in predvaja zvočna opozorila.

## Namestitev in Uporaba

### Namestitev
1. **Kloniranje repozitorija**:
   ```bash
   git clone https://github.com/uporabnik/FaceEyeDetectionApp.git
   cd FaceEyeDetectionApp
