import cv2 # Görüntü işleme için kütüphane.
import numpy as np 
from tensorflow.keras.models import load_model #model için kullanılan hazır fonksiyon.

# modeli yükler
model = load_model('emotion_detection_model.h5', compile=False)
# compile=False: Modelin sadece tahmin için kullanılacağını belirtir

# Modelin çıktı olarak vereceği 0-6 arası sayısal indislerin karşılık geldiği duygu isimleri.
duygu_listesi = ['Kizgin', 'Igrenme', 'Korku', 'Mutlu', 'Uzgun', 'Saskin', 'Dogal']

# OpenCV'nin hazır yüz tespit algoritmasınıyükler. 
yuztakibi = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


tahmin_yapılsın_mı = False
son_duygu = "" # Ekranda gösterilecek olan en son tahmin edilen duyguyu saklar.
buton_alanı = [20, 20, 170, 70] 
# Fare hareketlerini ve tıklamalarını dinleyen geri çağırma (callback) fonksiyonu.
def fare_olayı(event, x, y, flags, param):
    global tahmin_yapılsın_mı
    #farenin sol tuşu (EVENT_LBUTTONDOWN):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Eğer tıklanan farenin (x,y) koordinatları, belirlediğimiz buton alanının içindeyse:
        if buton_alanı[0] <= x <= buton_alanı[2] and buton_alanı[1] <= y <= buton_alanı[3]:
            tahmin_yapılsın_mı = True # Tahmin tetikleyicisini aktif et.

cv2.namedWindow("Duygu Analizi")
# fareyi takip etmek için kullanıldı
cv2.setMouseCallback("Duygu Analizi", fare_olayı)

kamera = cv2.VideoCapture(0)

while True:
    deger, kare = kamera.read() 
    if not deger: 
        break

    # görüntüyü griye çevirir
    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    
    
    # 1.3: Görüntüyü %30 küçülterek tarar. üü
    # 5: Bir bölgenin yüz sayılması için en az 5 komşu bölgeyle doğrulanması gerekir.
    yuzler = yuztakibi.detectMultiScale(gri, 1.3, 5)

    # butonu çiziyoruz
    cv2.rectangle(kare, (buton_alanı[0], buton_alanı[1]), (buton_alanı[2], buton_alanı[3]), (0, 200, 0))
    # Butonun üzerine "Tahmin Et" metnini yazar.
    cv2.putText(kare, "Tahmin Et", (buton_alanı[0]+15, buton_alanı[1]+35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Tespit edilen her bir yüz için döngü başlatır (x,y: koordinat; w,h: genişlik,yükseklik).
    for (x, y, w, h) in yuzler:
        # Yüzün etrafına yeşil bir çerçeve çizer.
        cv2.rectangle(kare, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Eğer butona tıklandıysa (tahmin_yapılsın_mı True ise):
        if tahmin_yapılsın_mı:
            yuz = gri[y:y+h, x:x+w] # Sadece yüzün olduğu bölgeyi alır.
            yuz = cv2.resize(yuz, (48, 48)) # 48*48pixel
            yuz = yuz.astype("float32") / 255.0 # normalize işlemi.
            yuz = np.expand_dims(yuz, axis=0) 
            yuz = np.expand_dims(yuz, axis=-1) 

            # verbose=0: İşlem sırasında konsola çıktı basmaz.
            tahmin = model.predict(yuz, verbose=0)
            sonuc = np.argmax(tahmin) # 7 olasılık arasından en yüksek olanın indexini alır.
            son_duygu = duygu_listesi[sonuc] # İndexi listedeki duygu adına çevirir.
            tahmin_yapılsın_mı = False #sürekli tahmin yapmamsı için tahmin sonrası işlemi durdurur.

        # Tahmin edilen duygu metnini, yüz çerçevesinin hemen üzerine yazar.
        cv2.putText(kare, son_duygu, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Duygu Analizi", kare)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release() 
cv2.destroyAllWindows() 