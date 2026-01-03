🎭 Gerçek Zamanlı Duygu Analizi (Emotion Detection)
Bu proje, bilgisayar kamerası üzerinden alınan görüntülerde yüz tespiti yaparak, tespit edilen yüzün duygusal durumunu (mutlu, üzgün, kızgın vb.) derin öğrenme modeli yardımıyla tahmin eder.
Kullanıcı, ekrandaki “Tahmin Et” butonuna tıkladığında sistem yalnızca tek bir tahmin yapar ve sonucu ekranda gösterir.
________________________________________
📌 Projenin Özellikleri
•	Gerçek zamanlı kamera görüntüsü
•	OpenCV ile yüz tespiti (Haar Cascade)
•	Eğitilmiş CNN modeli ile duygu tahmini
•	Fare tıklamasıyla kontrollü tahmin
•	7 farklı duygu sınıfı
•	Basit ve kullanıcı dostu arayüz
________________________________________
🎯 Tahmin Edilen Duygular
Model aşağıdaki 7 duygu sınıfını tahmin edebilir:
1.	Kızgın
2.	İğrenme
3.	Korku
4.	Mutlu
5.	Üzgün
6.	Şaşkın
7.	Doğal
________________________________________
🛠 Kullanılan Teknolojiler ve Kütüphaneler
•	Python 3
•	OpenCV (cv2) – Görüntü işleme ve yüz tespiti
•	NumPy – Sayısal işlemler
•	TensorFlow / Keras – Derin öğrenme modeli
________________________________________


▶️ Çalıştırma Adımları
1.	Gerekli kütüphaneleri yükleyin:
2.	pip install opencv-python numpy tensorflow
3.	Kamera erişimi olan bir bilgisayarda kodu çalıştırın:
4.	python emotion_detection.py
5.	Kamera açıldıktan sonra:
o	Yüzünüzü kameraya gösterin
o	“Tahmin Et” butonuna tıklayın
o	Algılanan duygu ekranda gösterilecektir
6.	Programdan çıkmak için:
o	Klavyeden q tuşuna basın
________________________________________
🧠 Çalışma Mantığı (Özet)
•	Kamera görüntüsü alınır
•	Görüntü gri tona çevrilir
•	Haar Cascade ile yüz tespiti yapılır
•	Kullanıcı butona tıklarsa:
o	Yüz 48x48 boyutuna getirilir
o	Normalize edilir
o	CNN modele gönderilir
o	En yüksek olasılığa sahip duygu ekrana yazdırılır
________________________________________
⚠️ Notlar
•	Model yalnızca tahmin (predict) amaçlı kullanıldığı için compile=False ayarlanmıştır
•	Sürekli tahmin yapılmaması için tahmin işlemi buton tıklamasıyla sınırlandırılmıştır
•	Daha iyi sonuçlar için iyi aydınlatılmış ortam önerilir
