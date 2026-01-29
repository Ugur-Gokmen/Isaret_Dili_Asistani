import cv2
import mediapipe as mp
import numpy as np
import os


GÖRÜNTÜ_YOLU = 'test_el.jpg'

if not os.path.exists(GÖRÜNTÜ_YOLU):
    print(f"HATA: '{GÖRÜNTÜ_YOLU}' dosyası bulunamadı. Lütfen bir resim dosyası koyun.")
    exit()


mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils


with mp_hands.Hands(
        static_image_mode=True,  # Resimler için en iyi modu seçer
        max_num_hands=1,  # Maksimum 1 el tespit et
        min_detection_confidence=0.5) as hands:  # Tespit güven aralığı

   
    image = cv2.imread(GÖRÜNTÜ_YOLU)

    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    results = hands.process(image_rgb)

   
    if results.multi_hand_landmarks:
       
        hand_landmarks = results.multi_hand_landmarks[0]

        
        print("El Başarıyla Tespit Edildi!")
        print("-" * 30)

       
        
        landmark_data = []

       
        for landmark in hand_landmarks.landmark:
            landmark_data.append([landmark.x, landmark.y, landmark.z])

        landmark_array = np.array(landmark_data)

      
        print(f"Kilit Noktası NumPy Dizisi Şekli: {landmark_array.shape}")
        print(f"Başlangıç Noktası (Bilek, Landmark 0) Koordinatları: {landmark_array[0]}")

        # # OPSİYONEL: Görüntü üzerine kilit noktalarını ve iskeleti çizme
        # annotated_image = image.copy()
        # mp_drawing.draw_landmarks(
        #     annotated_image,
        #     hand_landmarks,
        #     mp_hands.HAND_CONNECTIONS,
        #     mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        #     mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # # cv2.imshow('Tespit Edilen El', annotated_image)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()

    else:

        print("Görüntüde el tespit edilemedi.")
