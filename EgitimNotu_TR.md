# Ollama Cloud ile Açık Kaynak LLM'leri Donanım Yükseltmeden Denemek (Yeni Başlayanlar İçin Ders Notu)

Açık Kaynak büyük dil modelleri (LLM) dünyasında her gün yeni bir model ve yeni bir sürüm konuşuluyor. Bu çok iyi bir haber, çünkü artık "tek bir model"e
mahkum değiliz. Ama pratikte çoğu kişi aynı iki probleme takılıyor: (1) Büyük modeller yerel GPU VRAM'ine sığmıyor, (2) Birden fazla modeli denemek için
her seferinde devasa dosyalar indirmek, kurmak, silmek hem zaman hem de disk alanı tüketiyor. Kısacası, öğrenme ve deneme süreci daha en başta yorucu hale gelebiliyor.

Ben bu eğitimi tam da bu yüzden hazırladım: Açık Kaynak LLM'leri **Bilgisayarınıza** model dosyalarını indirip biriktirmeden de test edebilin.
Üstelik sadece "çalıştırma" değil, doğru model seçimi ve uygulamaya entegrasyon mantığını da anlayın. Bu yazı, YouTube'daki eğitimimin blog formatında,
ders notu kadar derinlemesine hazırlanmış versiyonu. Adım adım giderken, neden-sonuç ilişkisini de kurmaya çalışacağım.

Eğitim videosu (YouTube canlı yayın kaydı):
[https://youtube.com/live/QEZ8oF4A68k](https://youtube.com/live/QEZ8oF4A68k)

Kodlar ve dokümanlar (GitHub deposu):
[https://github.com/kmkarakaya/OllamaCloud](https://github.com/kmkarakaya/OllamaCloud)

Not: Bu yazıda mümkün oldukça Türkçe terimler kullanıyorum. Örneğin "Açık Kaynak" (open source), "bulut", "belirteç" (token), "akış" (stream) gibi. Kod ve komutlarda ise resmi isimler ve parametreler aynen kalıyor.

## Bu Ders Notunu Nasıl Kullanmalısın?

Bu yazı iki tip okuyucu için tasarlandı:

- LLM dünyasına yeni girenler: "Ben nereden başlayacağım, hangi kavramlar önemli?" diyenler.
- Uygulama geliştirmek isteyenler: "Ben Python ile bir şey kurmak istiyorum, model denemeyi hızlı hale getirmek istiyorum." diyenler.

Benim önerim şu: Önce kavramları ve karar mantığını oku. Sonra CLI ile bir modeli çalıştır. Ardından API ve Python bölümüne geç. En sonda repodaki iki örneği
(model karşılaştırma betiği ve Streamlit demo) çalıştırıp sonuçları yorumla. Bu sırayla ilerlersen, "ne yaptığını bilerek" ilerlemiş olursun.

## Temel Kavramlar: Modeller Neden Ağır? (VRAM, RAM, Disk, belirteç)

Bir modelin "büyük" veya "ağır" olması tek bir şeye bağlı değil. Yeni başlayanların kafasını en çok karıştıran nokta da bu. Ben burada üç kaynak üzerinden anlatıyorum:
**Disk**, **RAM** ve **VRAM**.

- **Disk**: Model dosyalarını indirip sakladığın yer. Birkaç model denemek bile onlarca GB yapabilir.
- **RAM**: CPU belleği. Bazı durumlarda (özellikle CPU ile çalışan senaryolarda) RAM sınırlayıcı olur.
- **VRAM**: GPU belleği. Büyük modellerin yerelde hızlı çalışmasında çoğu zaman asıl sınırlayıcı kaynak budur.

Parametre sayısı arttıkça (örneğin 20B, 120B gibi) modelin kapasitesi artabilir, ama kaynak ihtiyacı da artar. Ancak tek mesele parametre sayısı değildir.
Model çalışırken "KV cache" (kısaca bağlam belleği) de büyür. Bu cache, konuşma uzadıkça ve context window büyüdükçe artar. Bu yüzden "kısa bir soru" ile "uzun bir sohbet"
aynı modelde farklı kaynak tüketimi üretebilir.

Bir diğer kritik kavram: **belirteç**. Modelin okuduğu ve ürettiği metin parçaları belirteçlere bölünür. Bir prompt ne kadar uzunsa, o kadar fazla belirteç işlenir.
Bir konuşma geçmişi ne kadar uzunsa, modelin "tek seferde düşünmesi" için o kadar fazla belirteç taşınır. Bu hem süreyi (gecikme) hem de bellek ihtiyacını etkiler.
Bu yüzden iyi bir pratik: Modeli değerlendirirken sadece tek bir prompt değil, farklı uzunluklarda promptlar denemek.

Peki quantization (kuantizasyon) ne işe yarar? Çok basit anlatayım: Modeli daha düşük hassasiyetle temsil ederek daha az bellek kullanmasını sağlar.
Bu sayede bazı modeller yerelde çalışabilir hale gelir. Ama bazen kalite ve tutarlılık üzerinde etkisi olabilir. Bu nedenle model seçimi yaparken "en büyük model en iyisidir" gibi
basit bir kural yok. Senin senaryon için "yeterince iyi + yeterince hızlı + yönetilebilir maliyet" dengesi önemlidir.

Bu noktada Ollama Cloud'un değeri ortaya çıkıyor: İndirme ve donanım bariyerini azaltıp, senin hızlıca öğrenme ve deneme yapmanı sağlıyor.
Yani önce doğru modeli seçiyorsun, sonra yerel veya bulut stratejini buna göre planlıyorsun.

## KV Cache ve Context Window: Uzun Sohbet Neden Yavaşlar?

Yeni başlayanların en sık yaşadığı sürprizlerden biri şudur: Aynı model, kısa bir promptta çok hızlı cevap verirken; konuşma uzadıkça veya tek prompt çok uzayınca
bariz şekilde yavaşlar. Bunun temel nedeni "modelin geçmişi nasıl taşıdığı" ile ilgilidir. Dil modelleri, her yeni belirteç üretirken önceki belirteçlere dikkat (attention)
mekanizmasıyla bakar. Konuşma uzadıkça bakması gereken içerik artar; bu da hesaplama maliyetini ve bellek ihtiyacını yükseltir.

Bu işin bellek tarafındaki adı çoğu yerde *KV cache* olarak geçer. Çok basit anlatımla: Model, önceki belirteçlere ait bazı ara temsil bilgilerini (key/value)
cache'ler ve sonraki adımlarda bunları kullanır. Context window büyüdükçe bu cache de büyür. Yerelde GPU kullanıyorsan bu cache çoğu zaman VRAM'de tutulur ve
VRAM'in sınırlıysa uzun konuşmalarda daha çabuk sınıra dayanırsın. bulut'da bu sınır sende görünmez ama performans etkisini yine hissedebilirsin.

Bu yüzden ben model denemesinde şunu özellikle öneriyorum: Sadece "tek cümlelik bir soru" ile karar verme. Aynı modelde kısa prompt, orta prompt ve uzun prompt dene.
Bir de "konuşma geçmişi" senaryosu dene. Çünkü gerçek uygulamada çoğu zaman kullanıcı bir soru sorup çıkmıyor; takip sorusu soruyor, detay istiyor, format değiştiriyor.
Modelin bu akıştaki davranışı, tek seferlik cevap kadar önemlidir.

Pratik bir performans ipucu: Uygulamada konuşma geçmişini sınırsız büyütme. Gerekirse eski mesajları özetle, kritik noktaları "memory" gibi tek bir mesajda taşı.
RAG (retrieval-augmented generation) gibi teknikler de burada devreye girer: Her şeyi konuşma geçmişinde tutmak yerine, ilgili bilgiyi dışarıdan çekip prompta eklersin.
Bu hem maliyeti hem de gecikmeyi yönetmeyi kolaylaştırır.

## Ollama Cloud Nedir? (CLI ve API İki Yol)

Ollama Cloud, desteklenen modelleri bulut üzerinden çalıştırıp denemeni sağlayan bir altyapı. Senin tarafında iki "arayüz" var:
**Ollama CLI** ve **Ollama API**. Benim ders notu yaklaşımımda bunlar iki farklı hedefe hizmet eder:

- **CLI yolu**: Hızlı deneme ve hızlı kıyaslama. Prompt yaz, cevap al, model davranışını gör.
- **API yolu**: Uygulama entegrasyonu. Python ile web app, bot, servis veya otomasyon kur.

Resmi dokümanlar (okurken açık dursun):
[https://docs.ollama.com/cloud](https://docs.ollama.com/cloud),
[https://docs.ollama.com/api](https://docs.ollama.com/api)

Not: "Ücretsiz deneme" ve limitler dönemsel olarak değişebilir. En güncel bilgiyi resmi sayfalardan kontrol etmeni öneririm.

## Ne Zaman bulut, Ne Zaman Yerel?

bulut ve yerel kullanım birbirinin alternatifi değil; çoğu zaman birbirini tamamlar. Ben pratikte şöyle karar veriyorum:

- **bulut-öncelikli**: Model denemek, model seçmek, hızlı prototip yapmak, eğitim/demo üretmek.
- **Local-first**: Offline çalışma, çok sık istek, veri politikası nedeniyle tamamen yerel zorunluluk.

Eğer amacın "hangi model işimi görür?" sorusunu cevaplamaksa bulut yaklaşımı büyük hız kazandırır. Modeli seçtikten sonra yerelde koşmak istiyorsan,
o aşamada donanım yatırımı veya daha küçük/kuantize bir model seçimi mantıklı hale gelir. Yani benim için bulut çoğu zaman bir "hızlandırıcı katman".

## Model Deneme ve Karşılaştırma İçin Basit Bir Rubrik

Model denemek demek, aynı soruyu üç modele sormak demek değildir. Gerçek bir karşılaştırma yapmak için küçük bir rubrik gerekir.
Benim kullandığım basit rubrik şu başlıklardan oluşur:

- **Talimat takibi**: "Şu formatta yaz" dediğimde uyuyor mu?
- **Tutarlılık**: Aynı soruyu tekrar sorunca benzer kalite veriyor mu?
- **Yapı ve okunabilirlik**: Cevap düzenli mi, maddeleme iyi mi?
- **Doğruluk**: Bariz hatalar yapıyor mu, uydurma eğilimi var mı?
- **Hız**: Aynı prompt için yanıt süresi kabul edilebilir mi?
- **Dil uyumu**: Türkçe sorularda Türkçe kalitesi yeterli mi?

Bu rubriği kullanarak "model seçimi"ni hızlandırabilirsin. Örneğin eğitimde kullandığım yaklaşım: Aynı konuda 2-3 farklı prompt tipi hazırla,
sonra her modeli bu prompt setiyle test et. Böylece tek bir cevaba göre karar vermezsin. Özellikle "format zorlayan" promptlar modelin talimat takibini çok net gösterir.

Bu rubriği repodaki `compare_models.py` ile otomatikleştirmeye başladık. betik mükemmel bir kıyaslama değil, ama doğru düşünme biçimini öğretir:
Önce yapılandırılmış görevler, sonra gerçek çıktı incelemesi.

## Prompt Tasarımı İpuçları: Modeli Doğru Sınamak

Bir modelin "iyi" veya "kötü" olduğuna karar vermek için önce doğru prompt yazmak gerekir. Çünkü LLM'ler, soruyu nasıl sorduğuna göre çok farklı davranabilir.
Benim ders notu şeklinde önerdiğim basit yaklaşım: Promptu dört parçaya böl ve her parçayı net yaz.

- **Rol**: Modelden nasıl davranmasını istiyorsun? (Örn: "Bir eğitmen gibi anlat.")
- **Hedef**: Tam olarak ne istiyorsun? (Örn: "RAG'i 5 maddeyle açıkla.")
- **Kısıt**: Uzunluk, dil, format, ton gibi sınırlar. (Örn: "Her madde 1 cümle olsun.")
- **Çıktı formatı**: Madde madde mi, JSON mu, tablo mu? (Örn: "Başlık + maddeler.")

Karşılaştırma yaparken de aynı konsept geçerli: Aynı rubrikle ölçmek istiyorsan, modellerin hepsine aynı formatı zorlayan promptları ver.
Örneğin Türkçe üreteceksen mutlaka Türkçe prompt seti hazırla. Kod yazdıracaksan "basit kod", "hata ayıklama", "refactor" gibi farklı kod görevleri ekle.
Bu sayede tek bir promptta parlayan ama diğer görevlerde dağılan modelleri erken fark edersin.

Bir başka pratik ipucu: Deneme promptlarını "gerçek iş" promptlarından ayır. Deneme promptu daha kısa ve daha kontrollü olur; gerçek iş promptu ise daha karmaşık ve
daha çok bağlam içerir. Bu yüzden karar verirken iki tür promptu da kullan. Deneme promptu ile ele, gerçek iş promptu ile doğrula.

## Güvenlik ve İyi Pratikler: API Key ile Çalışmak

API key bir şifre gibi düşünülmeli. En sık yapılan hata, API key'i koda gömmek veya depo içinde paylaşmak. Benim önerim:
API key'i bir ortam değişkeni olarak tanımla ve uygulamada oradan oku.

Resmi yetkilendirme dokümanı:
[https://docs.ollama.com/api/authentication](https://docs.ollama.com/api/authentication)

Windows PowerShell tarafında, geçerli oturum için örnek:
```
$env:OLLAMA_API_KEY = "your_api_key_here"
```

Bu şekilde key'i kaynak koduna yazmadan, hem CLI tarafında hem de Python tarafında kullanabilirsin. Eğitim içeriklerinde özellikle şunu vurguluyorum:
Key yönetimi "küçük bir detay" değil; ileride üretim ortamına gittiğinde en kritik alışkanlıklardan biri olacak.

## CLI ile Başlamak: İlk Deneme Akışı (Neden Bu Komutlar?)

Önce Ollama'nın Windows kurulumu gerekiyor. Resmi Windows kurulum sayfası:
[https://docs.ollama.com/windows](https://docs.ollama.com/windows)

Kurulumdan sonra amaç şu: bulut tarafına giriş yap ve bir modeli terminalde çalıştır. Temel akış:
```
ollama --version
ollama signin
ollama run gpt-oss:120b-cloud
```

Burada `ollama signin` kritik. bulut erişimi gerektiren işlemlerde yerel CLI'nin kimlik doğrulamasını yapmış oluyorsun.
Sonra `ollama run` ile modeli interaktif moda alıyorsun. Bu noktada önemli pratik: İlk denemelerde kısa promptlarla başla, sonra uzun promptlara geç.
Çünkü uzun prompt, hem hız hem de bağlam yönetimi açısından modeli daha fazla zorlar.

Bir diğer önemli not: bulut model isimleri CLI tarafında bazen `-cloud` ekiyle gelir. API tarafında ise model adları farklı görünebilir.
Bu yüzden ben API entegrasyonlarında "model adını hardcode etme" alışkanlığını bırakmanı öneriyorum. Bunun çözümü bir sonraki bölümde: `/api/tags`.

## Direkt API Yolu: /api/tags ile Model Keşfi, /api/chat ile Cevap

Uygulama geliştireceksen API tarafı kritik. Ben burada iki endpointi "temel taş" gibi görüyorum:

- **/api/tags**: Hangi modeller erişilebilir? Bu sorunun cevabı.
- **/api/chat**: Mesaj gönderip cevap almak.

Endpoint dokümanları:
[https://docs.ollama.com/api/tags](https://docs.ollama.com/api/tags),
[https://docs.ollama.com/api/chat](https://docs.ollama.com/api/chat)

Benim pratik yaklaşımım: Model adını "varsayma". Önce `/api/tags` ile listeyi çek, sonra uygulamanda o listeden seçim yaptır.
Bu sayede bir model erişilebilir değilse uygulama kırılmaz; sadece seçeneklerde görünmez.

Minimal bir `curl` örneği (PowerShell'de):
```
curl https://ollama.com/api/tags `
  -H "Authorization: Bearer $env:OLLAMA_API_KEY"
```

sohbet tarafında `stream` parametresi önemli bir kavram. `stream=false` dersen tek seferde cevap alırsın.
`stream=true` dersen cevap parça parça gelir. Web uygulamalarında akış kullanıcı deneyimini iyileştirir,
ama kodu biraz daha dikkatli yazmayı gerektirir. Eğitimde ben önce `stream=false` ile mantığı kurup, sonra akışa geçmeni öneriyorum.

## Python ile Entegrasyon: Mantığı Bir Kez Kur, Her Yerde Kullan

Python tarafında benim önerim şu: Önce en küçük "çalışıyor mu?" prototipini kur. Sonra aynı mantığı web uygulamasına taşı.
Ollama Cloud Python dokümanı burada:
[https://docs.ollama.com/cloud#python](https://docs.ollama.com/cloud#python)

Kullandığımız Python paketi:
[https://github.com/ollama/ollama-python](https://github.com/ollama/ollama-python)

Python tarafında temel fikir:

- İstemciyi `host="https://ollama.com"` ile bulut'a yönlendir
- Header'a `Authorization: Bearer ...` koy
- `chat()` çağrısında model ve mesajları gönder

Minimal örnek:
```
import os
from ollama import Client

client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
)

resp = client.chat(
    model="gpt-oss:120b",
    messages=[{"role": "user", "content": "RAG nedir? 5 maddeyle anlat."}],
    stream=False,
)

print(resp["message"]["content"])
```

Burada mesaj formatı çok önemli. Mesajlar bir liste ve her mesajın `role` alanı var (örneğin `user`).
Gerçek uygulamada konuşma geçmişi tutarsın: Önceki mesajları da yeni promptla birlikte gönderirsin. Bu sayede model bağlamı "hatırlar".
Ama şu riski unutma: Konuşma geçmişi uzadıkça belirteç sayısı artar; bu da gecikmeyi artırabilir. Bu yüzden pratik bir yöntem:
Geçmişi sınırlamak veya kritik noktaları özetleyip "memory" gibi göndermek.

Hata senaryolarını ders notu gibi düşünelim:

- **401 Unauthorized**: API key yanlış veya header formatı hatalı. İlk kontrol: ortam değişkeni + auth dokümanı.
- **404 / model not found**: Model adı erişilebilir değil; önce `/api/tags` ile listeyi kontrol et.
- **Timeout/slow**: Çok büyük model veya yoğunluk; daha küçük modelle dene, istekleri azalt, akış kullan.

Bu üç hata tipi, pratikte en sık görülenler. Eğitimde de özellikle "önce tags, sonra sohbet" mantığını vurguluyorum.

## depo Walkthrough: İki Bağımsız Örnekle Öğrenmeyi Hızlandırmak

Eğitimi izleyip "tamam anladım" demek kolay. Asıl değer, iki örneği çalıştırıp sonuçları yorumlamakta.
depo içinde üç kritik dosya var: **requirements.txt**, **compare_models.py**, **app.py**.
Bu üçü birlikte "deneme -> karşılaştırma -> demo" akışını tamamlıyor.

Önce bağımlılıkları kurmak ve örnekleri çalıştırmak için pratik komutlar:
```
pip install -r requirements.txt
python compare_models.py
streamlit run app.py
```

Şimdi iki örneği tek tek ders notu gibi inceleyelim.

**compare_models.py** ne yapıyor?
Bu betik "model seçimi" problemine pratik bir yaklaşım getiriyor. İçindeki iki yapı çok önemli:

- **MODELS**: Karşılaştırılacak model listesi (ör. gpt-oss:20b, gpt-oss:120b, qwen3-coder:480b)
- **TASKS**: Modelin cevap vermesini istediğin yapılandırılmış görevler

Scriptin yaklaşımı şu: Her modele aynı görevleri sorar, yanıt süresini ölçer ve basit bir kalite skoru çıkarır.
Bu skor bir hakem değil; bir "erken sinyal". Skorun mantığı dosyada açık:

- Anahtar kelime kapsama skoru (0-60): Görev için kritik kelimeleri geçiriyor mu?
- Format skoru (0-40): Maddeleme var mı, satır sayısı makul mü, yanıt uzunluğu yeterli mi?

Skorun amacı şu: Hızlıca "bu model talimatları takip ediyor mu?" sorusuna yaklaşmak. Ama karar verirken mutlaka raporu okuyacaksın.
betik, tüm çıktıları `model_comparison_report.md` dosyasına yazar. Bu dosyayı açıp yanıtları yan yana okumak, gerçek kalite farkını görmenin en iyi yoludur.

Bu betiği kendi senaryona uyarlamak için pratik öneriler:

- Kendi işine uygun 3-5 görev ekle (Türkçe özetleme, kod üretimi, e-posta yazma, hata ayıklama vb.).
- MODELS listesini `/api/tags` ile gördüğün erişilebilir modellere göre güncelle.
- Skorun "tek ölçüt" olmadığını unutma; raporu mutlaka oku.

Şimdi ikinci örnek: **app.py**.
Bu dosya Streamlit ile basit bir web arayüzü kuruyor. Bu demo, özellikle öğrenme ve manuel test için çok değerli.
Çünkü CLI'de hızlı denersin ama bazen promptları düzenlemek, farklı formatları denemek, çıktıyı kopyalayıp incelemek tarayıcıda daha rahattır.

Uygulamanın akışı:

- API key'i ortamdan alır veya kullanıcıdan ister.
- bulut'dan model listesini çeker ve seçim sunar.
- Prompt gönderilir, cevap ekranda gösterilir.

Bu yapının asıl öğretici tarafı şudur: Modeli "hardcode" etmek yerine listeden seçtirmek. Böylece erişilebilir modeller değişse bile uygulama dayanıklı olur.
Bu, üretim sistemlerinde de çok işe yarayan bir alışkanlıktır.

Uygulamayı geliştirmek istersen, başlangıç için birkaç fikir:

- Konuşma geçmişi (sohbet geçmişi) ekle: Önceki mesajları saklayıp tekrar gönder.
- akış yanıt ekle: Cevap yazılıyor gibi görünsün.
- Prompt şablonları ekle: "özetle / maddele / kod yaz" gibi hazır butonlar.
- Çıktıyı kaydet: Prompt ve cevapları logla, sonra analiz et.

## Sık Yapılan Hatalar ve Pratik Çözümler

Bu bölüm gerçek hayatın özeti: Hata alırsın ve çözersin. Ben en sık şunları görüyorum:

- **401 Unauthorized**: API key yok, yanlış veya header formatı hatalı. Önce [Authentication](https://docs.ollama.com/api/authentication) dokümanına bak, sonra ortam değişkenini kontrol et.
- **Model bulunamadı**: Model adı erişilebilir değil. İlk adım: [/api/tags](https://docs.ollama.com/api/tags) ile modelleri listele ve doğru adı seç.
- **Çok yavaş yanıt**: Büyük model veya yoğunluk olabilir. Daha küçük model dene, promptu kısalt, akış kullanmayı düşün.
- **Virtualenv karışıklığı**: Farklı Python ortamına kurulum yapılıyor. En temiz yöntem: venv aç, sonra `pip install -r requirements.txt`.
- **Türkçe karakterler bozuk görünüyor**: Kopyalama yaparken HTML editör modu dışında yapıştırmış olabilirsin. Blogger'da HTML görünümünde yapıştırmak genelde en sorunsuz yoldur.

Bu hataların ortak mesajı şu: Önce bağlantı ve yetkiyi doğrula, sonra model listesini doğrula, en son uygulama kodunu kurcala.
Bu sırayla ilerlemek zamandan tasarruf ettirir.

## Sık Sorulan Sorular (SSS)

Bu eğitimden sonra en çok gelen soruları kısa ve net cevaplayayım. Buradaki amaç "ezber" değil; doğru kontrol noktalarını öğretmek.

- **Ollama Cloud ücretsiz mi?** Çoğu kişi bulutu "hızlı deneme" için kullanıyor. Ancak limit/politikalar zaman içinde değişebilir. En güncel bilgi için [bulut](https://docs.ollama.com/cloud) sayfasını kontrol et.
- **Hangi modeller var, hangisini seçmeliyim?** "Şu model kesin var" diye varsayma. Önce [/api/tags](https://docs.ollama.com/api/tags) ile listeyi gör. Seçimi rubrikle yap: talimat takibi, tutarlılık, hız, Türkçe kalitesi ve senin senaryona uygunluk.
- **CLI'de -bulut var ama API'de yok, neden?** CLI ve API tarafında isimlendirme farklı görünebilir. Uygulama tarafında modeli listeden seçtirmen bu sorunu pratikte çözer.
- **ollama signin ile API key aynı şey mi?** Hayır. `ollama signin` daha çok CLI akışı için oturum açma mantığıdır. API tarafında ise genelde API key + Bearer header ile çalışırsın. Detay için [Authentication](https://docs.ollama.com/api/authentication) sayfasına bak.
- **Veri gizliliği açısından neye dikkat etmeliyim?** Kural basit: Hassas veriyi göndermeden önce mutlaka politika/şartları oku ve kurumunun kurallarına göre hareket et. Eğitim ve demo sırasında gerçek müşteri verisi yerine sentetik/anonim veri kullanmak iyi bir alışkanlıktır.
- **Türkçe karakterler bozuk görünürse?** Genelde iki sebep olur: dosya encoding'i veya Blogger'a yapıştırma modu. Bu dosya UTF-8 olarak hazırlanmıştır. Blogger'da HTML modunda yapıştırmak çoğu zaman sorunu çözer.

## Mini Çalışma: Kendi Deneme Setini Oluştur

Bu yazıyı gerçekten değerli kılmak için sana küçük bir "ödev" seti bırakmak istiyorum. Eğer bir modeli seçmek istiyorsan, şu üç tip prompt hazırla:

- **Format testi**: "5 maddeyle anlat, her madde 1 cümle olsun" gibi.
- **Akıl yürütme testi**: "Adım adım açıkla" gibi.
- **Senaryo testi**: "Benim işim şu, bana bir taslak çıkar" gibi.

Sonra aynı üç promptu üç farklı modelde dene. Karşılaştırma yaparken sadece "doğru mu?" diye bakma; rubriğe geri dön:
Talimat takibi, tutarlılık, okunabilirlik ve hız. Bu küçük alışkanlık, öğrenme sürecini hızlandırır ve model seçimini daha sağlam yapar.

## Benim Önerdiğim Deneme Stratejisi

Eğer bir proje için model seçeceksem ben genelde şu akışla ilerliyorum:

- Önce 2-3 aday model belirlerim (biri hızlı/küçük, biri büyük/kaliteli, biri kod odaklı vb.).
- `compare_models.py` ile hızlı karşılaştırma yaparım.
- Rapor dosyasından cevapları okur, talimat takibi ve tutarlılığa bakarım.
- Seçtiğim 1-2 modeli `app.py` ile manuel prompt testine sokarım.
- Sonra uygulama entegrasyonuna geçerim (Python client ile).

Bu akışın güzelliği şu: Model seçimi "hissiyat" değil, kısa ama sistematik bir deneme sürecine dayanıyor. Üstelik bunu donanım yatırımı yapmadan başlatabiliyorsun.

## Videoyu ve Kodları Burada Bulabilirsin

YouTube eğitim kaydı:
[https://youtube.com/live/QEZ8oF4A68k](https://youtube.com/live/QEZ8oF4A68k)

GitHub deposu (README ve örnek kodlar):
[https://github.com/kmkarakaya/OllamaCloud](https://github.com/kmkarakaya/OllamaCloud)

Resmi kaynaklar:

- [Ollama Cloud](https://docs.ollama.com/cloud)
- [Ollama Cloud Python](https://docs.ollama.com/cloud#python)
- [Ollama Windows](https://docs.ollama.com/windows)
- [Ollama API](https://docs.ollama.com/api)
- [API Authentication](https://docs.ollama.com/api/authentication)
- [API /tags](https://docs.ollama.com/api/tags)
- [API /chat](https://docs.ollama.com/api/chat)
- [ollama-python](https://github.com/ollama/ollama-python)

## Kapanış

Ben bu eğitimi, "donanımım yetmiyor" bahanesinin seni açık kaynak LLM denemekten uzaklaştırmaması için hazırladım. Ollama Cloud ile hem farklı modelleri
hızlıca deneyebilir, hem de seçtiğin modeli Python tarafında gerçek bir uygulamaya bağlayabilirsin.

Eğer sen de açık kaynak LLM'leri hızlıca denemek, karşılaştırma yapmak ve pratik bir akış kurmak istiyorsan videoyu izle, depo içindeki örnekleri çalıştır,
sonra kendi senaryona uyarlayıp geliştirmeye başla. Benzer içerikler için **Murat Karakaya Akademi**'yi takip etmeyi unutma.
