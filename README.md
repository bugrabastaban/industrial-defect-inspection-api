Industrial Defect Inspection API
Bu proje, Sakarya Üniversitesi Bilgisayar Mühendisliği son sınıf öğrencisi olarak, akademik süreç boyunca edinilen teorik altyapıyı "uçtan uca çalışan ve canlıya alınmış (production-ready) bir mühendislik sistemine nasıl dönüştürebilirim?" sorusuna odaklanarak geliştirdiğim bir portfolyo çalışmasıdır.

Sistemi yalnızca yerel geliştirme ortamı (localhost) ile sınırlı tutmak yerine; bilgisayarlı görü (computer vision), asenkron backend mimarisi, ilişkisel veritabanı yönetimi ve bulut dağıtımı (deployment) süreçlerini endüstri standartlarında, entegre bir çatı altında birleştirmeyi hedefledim.

Projenin temel amacı; endüstriyel üretim hatlarında meydana gelebilecek yüzey kusurlarını (çizik, kılcal çatlak, çukurlaşma vb.) gerçek zamanlı olarak tespit etmek, bu hatalı üretimleri MLOps prensiplerine uygun şekilde nesne depolama katmanında loglamak ve analiz sonuçlarına ait meta verileri bir SQL veritabanında yapılandırılmış olarak saklamaktır.

Proje Mimarisi ve Akış
Sistem temelde şu döngüyle çalışır:

Üretim bandından (istemciden) gelen hatalı parça görseli API'ye iletilir.

Özel eğitilmiş YOLOv8 modeli, görsel üzerindeki kusurları ve güven skorlarını milisaniyeler içinde tespit eder.

FastAPI, tespit edilen hataların koordinatlarını çıkarır.

MLOps mantığıyla görsel işlenerek diskte (S3 simülasyonu) arşivlenir.

Hatanın türü, güven skoru, dosya yolu ve tarihi SQLite veritabanına kaydedilir.

İstemciye (Postman) detaylı bir JSON formatında geri dönüş yapılır.

Kullanılan Teknolojiler
Yapay Zeka: Ultralytics YOLOv8 (NEU-DET veri seti ile eğitildi)

Backend: FastAPI, Uvicorn, Python 3.10+

Veritabanı: SQLAlchemy (ORM), SQLite

Görüntü İşleme: OpenCV, Pillow

Dağıtım (Deployment): Docker, Hugging Face Spaces

Test Aracı: Postman

Canlı Demo
Proje şu an Hugging Face Spaces üzerinde, Docker konteyneri içerisinde 7/24 canlı olarak çalışmaktadır. Kendi bilgisayarınıza hiçbir şey indirmeden doğrudan Postman üzerinden test edebilirsiniz.

Canlı API URL: https://bugrabastaban-industrial-project-space.hf.space

Postman ile Nasıl Test Edilir?
Geliştirme sürecinde API testleri için Swagger yerine daha kapsamlı senaryolar sunan Postman kullanılmıştır. Canlı sistemi test etmek için aşağıdaki adımları izleyebilirsiniz:

Görüntü Analizi ve Veritabanı Kaydı (POST İsteği)

Endpoint: /predict

Metot: POST

Body: form-data

Key: file (Tür: File olarak seçilmeli)

Value: Bilgisayarınızdan seçeceğiniz herhangi bir endüstriyel yüzey görseli.

Veritabanı Loglarını Görüntüleme (GET İsteği)

Endpoint: /logs

Metot: GET
Bu uç noktaya istek attığınızda, bugüne kadar veritabanına işlenmiş tüm tespit loglarını ve meta verileri en yeniden en eskiye doğru sıralı bir JSON dizisi olarak alabilirsiniz.
