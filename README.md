Harika, veri yapıları ve State (durum) yönetimi, LangGraph mimarisinin kalbidir. Bu yapıların neyi tuttuğunu ve neden var olduğunu açıklayan yorumları aşağıda hazırladım.

Bunları ilgili class tanımlarının hemen üzerine yapıştırabilirsin.

### `ErrorCorrection`

```python
# Hata düzeltme sürecinde LLM'den beklenen yapılandırılmış çıktıyı tanımlayan Pydantic modelidir. 
# Bir hata oluştuğunda, hatanın analizi sonrası sistemin hangi adıma (node) yönleneceğini (next_node) 
# ve hatayı gidermek için oluşturulan düzeltilmiş sorguyu (corrected_user_query) tutar. LLM'in 
# serbest metin yerine bu formatta yanıt vermesi, hata yönetiminin (error handling) programatik 
# olarak işlenebilmesini ve döngünün doğru yönlendirilmesini sağlar.
class ErrorCorrection(BaseModel):

```

### `ValidationResult`

```python
# Üretilen kodun veya analiz sonucunun doğruluk kontrolü (validation) sırasında LLM'in çıktısını 
# standartlaştıran modeldir. Analizin doğruluğu (is_correct), verilen güven puanı (accuracy_score) 
# ve varsa hataların nedenleri (feedback) bu yapıda tutulur. Sistemin sonucu kabul edip etmeyeceğine 
# veya yeniden deneme (retry) yapıp yapmayacağına karar vermesini sağlayan kritik metrikleri içerir.
class ValidationResult(BaseModel):

```

### `QueryEnhancement`

```python
# Doğrulama aşamasından gelen olumsuz geri bildirimlerle orijinal sorgunun iyileştirilmesi (refinement) 
# gerektiğinde kullanılan yapıdır. LLM'in, önceki hatalardan ders çıkararak oluşturduğu daha detaylı, 
# teknik ipuçları içeren ve zenginleştirilmiş yeni sorguyu (enhanced_query) kapsar. Bu yapı, 
# "kendi kendini düzelten" (self-correcting) döngünün bir sonraki denemede daha başarılı olmasını hedefler.
class QueryEnhancement(BaseModel):

```

### `PythonExecutionResult`

```python
# Python kodu çalıştırıldıktan sonra elde edilen ham sonuçların ve durum bilgisinin paketlendiği sınıftır. 
# İşlemin başarılı olup olmadığını, çalıştırılan kodu, oluşan DataFrame nesnesini veya varsa hata 
# mesajlarını tek bir çatı altında toplar. Bu yapı, ham verinin (örneğin pandas DataFrame) sistem 
# içinde güvenli bir şekilde taşınmasını ve son kullanıcıya gösterilecek formatın ayrıştırılmasını kolaylaştırır.
class PythonExecutionResult(BaseModel):

```

### `DataRoutingDecision`

```python
# Kullanıcı sorgusunun analiz edildikten sonra hangi iş akışına (workflow) yönlendirileceğini 
# belirleyen karar yapısıdır. LLM'in sorguyu "sadece bilgi", "kod üretimi" veya "doğrudan çalıştırma" 
# gibi kategorilere ayırmasını ve bu kararın arkasındaki mantığı (reasoning) tutar. Yönlendirme 
# düğümünün (routing node) çıktısının kesin, pars edilebilir ve iş akışını yönetebilir olmasını garanti eder.
class DataRoutingDecision(BaseModel):

```

### `DataAnalysisState`

```python
# LangGraph iş akışı boyunca düğümler (nodes) arasında taşınan ve paylaşılan merkezi veri yapısıdır (State). 
# Kullanıcının sorgusu, üretilen kodlar, veri bilgileri, hata mesajları, tekrar sayıları ve doğrulama 
# sonuçları gibi sürecin tüm anlık durumunu ("context") üzerinde tutar. Her bir düğüm bu sözlüğü 
# okuyarak ne yapacağına karar verir ve işlem sonucunu yine bu sözlüğe yazarak bir sonraki adıma aktarır; 
# yani grafın hafızasıdır.
class DataAnalysisState(TypedDict):

```


PANDAS
---

### `__init__`

```python
    # Bu kurucu metot (constructor), DataAnalysisAgent sınıfını başlatarak dil modelini (LLM), 
    # analiz edilecek veri çerçevesini (DataFrame) ve opsiyonel döküman işleyicileri yükler. 
    # Analiz süreci için gerekli olan maksimum tekrar deneme sayıları (retry limits) ve bağlam 
    # bulucu (context finder) gibi yardımcı araçların konfigürasyonunu yapar. Eğer harici bir 
    # döküman yolu veya sütun bilgisi sağlanmışsa, bu verileri işleyerek ajanın kullanımına hazır 
    # hale getirir. Son olarak, LangGraph tabanlı iş akışını (self.app) oluşturarak ajanı sorgu almaya hazır duruma getirir.
    def __init__(self, llm: ChatGroq, df: pd.DataFrame, doc_path: str = None, column_info_path: str = None):

```

### `_llm_routing_node`

```python
    # Bu fonksiyon, kullanıcının girdiği sorguyu analiz ederek sistemin izlemesi gereken en uygun 
    # iş akışını belirleyen akıllı bir karar mekanizmasıdır. LLM'i kullanarak sorguyu "sadece bilgi alma", 
    # "kod üretme" veya "kod çalıştırma" gibi kategorilere ayırır. Bu sınıflandırma sayesinde, 
    # gereksiz işlem adımlarının önüne geçilir ve sistem kaynakları verimli kullanılır. Fonksiyonun 
    # çıktısı, grafın bir sonraki adımda hangi düğüme (node) gideceğini belirleyen yapılandırılmış bir yönlendirme kararıdır.
    def _llm_routing_node(self, state: DataAnalysisState) -> DataAnalysisState:

```

### `get_data_info`

```python
    # Bu metot, analiz edilecek DataFrame hakkında yapısal bilgileri (satır/sütun sayısı), veri tiplerini 
    # ve örnek satırları toplayarak LLM için anlaşılır bir özet oluşturur. Eğer sistemde yüklü bir döküman 
    # veya sütun açıklaması varsa, RAG (Retrieval-Augmented Generation) benzeri bir yapıyla ilgili bağlamı 
    # da bu özete ekler. Temel amacı, kod üretimi aşamasında LLM'in veri setini "görmesini" sağlamak ve 
    # halüsinasyon görme ihtimalini azaltmaktır. Bu bilgiler 'state' sözlüğüne kaydedilerek diğer düğümlerin kullanımına sunulur.
    def get_data_info(self, state: DataAnalysisState) -> DataAnalysisState:

```

### `generate_pandas_code`

```python
    # Bu fonksiyon, toplanan veri bilgilerini ve kullanıcının isteğini kullanarak çalıştırılabilir, 
    # temiz bir Pandas kodu üretir. LLM'e özel bir prompt göndererek, sadece JSON formatında ve 
    # 'df' değişkenini temel alan, sonucu 'result' değişkenine atayan bir çözüm oluşturmasını ister. 
    # Üretilen kodun güvenli olması, gereksiz import içermemesi ve söz dizimi kurallarına uyması 
    # hedeflenir. Kod üretimi başarılı olursa, bu kodu state yapısına ekleyerek bir sonraki çalıştırma aşamasına hazırlar.
    def generate_pandas_code(self, state: DataAnalysisState) -> DataAnalysisState:

```

### `execute_python_code`

```python
    # Bu metot, bir önceki adımda üretilen Pandas kodunu izole edilmiş ve güvenli bir Python ortamında çalıştırır. 
    # 'exec' fonksiyonunu kullanarak kodu yürütür ve ortaya çıkan 'result' değişkenini yakalayarak işler; 
    # eğer sonuç bir DataFrame ise formatlayarak okunabilir bir metne dönüştürür. Kodun tehlikeli işlemler 
    # (dosya silme vb.) içermediğinden emin olmak için temel güvenlik kontrollerinden geçirir. Çalıştırma 
    # sırasında bir hata oluşursa, bu hatayı yakalayarak hata giderme (error handling) mekanizmasını tetikler.
    def execute_python_code(self, state: DataAnalysisState) -> DataAnalysisState:

```

### `validate_result`

```python
    # Bu fonksiyon, üretilen sonucun kullanıcının orijinal sorusuna gerçekten cevap verip vermediğini 
    # denetleyen bir kalite kontrol ve "eleştirmen" aşamasıdır. LLM'i bir hakem gibi kullanarak, 
    # kodun çıktısını mantıksal doğruluk ve tamlık açısından puanlar (0.0 - 1.0 arası). Eğer sonuç 
    # yetersizse veya hatalıysa, sistemin neden başarısız olduğuna dair detaylı geri bildirim (feedback) 
    # ve iyileştirme önerileri oluşturur. Bu geri bildirimler, bir sonraki döngüde sorgunun iyileştirilmesi için kullanılır.
    def validate_result(self, state: DataAnalysisState) -> DataAnalysisState:

```

### `enhance_query_with_feedback`

```python
    # Bu metot, doğrulama aşamasından gelen olumsuz geri bildirimleri kullanarak orijinal kullanıcı 
    # sorgusunu daha açık ve teknik açıdan yönlendirici hale getirir. Hatalı sonuçların nedenlerini 
    # analiz eder ve LLM'in bir sonraki denemede doğru kodu yazabilmesi için "ipuçları" içeren yeni 
    # bir prompt oluşturur. Örneğin, yanlış sütun adı kullanıldıysa, bu fonksiyon sorguya "Şu sütunu 
    # kullanma, doğrusu budur" gibi bir talimat ekler. Böylece sistem kendi kendini düzelterek (self-correction) doğru sonuca ulaşmaya çalışır.
    def enhance_query_with_feedback(self, state: DataAnalysisState) -> DataAnalysisState:

```

### `error_handling`

```python
    # Bu fonksiyon, iş akışının herhangi bir noktasında (kod üretimi, çalıştırma vb.) bir hata oluştuğunda 
    # devreye giren merkezi bir hata yönetim birimidir. Hata sayacını kontrol ederek sistemin sonsuz 
    # döngüye girmesini engeller ve maksimum tekrar sayısına ulaşılıp ulaşılmadığını denetler. Hatanın 
    # türüne ve karmaşıklığına bağlı olarak, durumu ya basit bir hata mesajıyla sonlandırır ya da daha 
    # akıllıca bir düzeltme yapması için LLM tabanlı hata düzeltme modülünü (_llm_error_correction) çağırır.
    def error_handling(self, state: DataAnalysisState) -> DataAnalysisState:

```

### `_llm_error_correction`

```python
    # Bu metot, alınan hata mesajını (Traceback vb.) analiz ederek hatanın kök nedenini (örneğin; syntax hatası, 
    # yanlış sütun adı, veri tipi uyuşmazlığı) tespit eder. Sadece hatayı raporlamakla kalmaz, aynı zamanda 
    # hatayı düzeltecek teknik talimatları içeren yeni ve düzeltilmiş bir kullanıcı sorgusu (corrected_user_query) 
    # oluşturur. Sistemin hangi adıma (node) geri dönmesi gerektiğine karar vererek (örneğin; tekrar kod üret 
    # veya tekrar veri bilgisi al) akışı dinamik olarak yönlendirir ve otonom hata çözümünü sağlar.
    def _llm_error_correction(self, state: DataAnalysisState) -> DataAnalysisState:

```

### `_route_condition`

```python
    # Bu yardımcı fonksiyon, graf içindeki düğümlerin (nodes) durumuna bakarak akışın bir sonraki adımda 
    # nereye gitmesi gerektiğini belirleyen mantıksal koşulları (conditional logic) içerir. 'state' içindeki 
    # verilere (kod üretildi mi, hata var mı, doğrulama geçti mi vb.) bakarak dinamik bir yol haritası çizer. 
    # Örneğin, kod üretildiyse çalıştırmaya, hata varsa hata yönetimine, sonuç doğrulandıysa bitişe (END) 
    # yönlendirme yapar. LangGraph yapısının "Conditional Edges" mantığını yöneten temel fonksiyondur.
    def _route_condition(self, state: DataAnalysisState) -> Literal["enhance_query_with_feedback","validate_result","get_data_info", "generate_pandas_code", "execute_python_code", "error_handling", "END"]:

```

### `_build_graph`

```python
    # Bu fonksiyon, tüm veri analiz sürecini yöneten LangGraph durum makinesini (StateGraph) inşa eder. 
    # Düğümleri (nodes) tanımlar, giriş noktasını belirler ve düğümler arasındaki koşullu geçişleri (edges) 
    # kurarak iş akışının iskeletini oluşturur. Hata durumunda nereye gidileceği, doğrulama sonrası 
    # döngülerin nasıl çalışacağı gibi tüm senaryoları birbirine bağlar. Son olarak, konuşma geçmişini 
    # ve durumu hatırlayabilmek için bir bellek (MemorySaver) ekleyerek grafı derler (compile).
    def _build_graph(self) -> StateGraph:

```

### `process`

```python
    # Bu metot, dış dünyadan (kullanıcıdan) gelen isteği alıp LangGraph iş akışını başlatan ana giriş 
    # kapısıdır (entry point). Kullanıcının sorgusunu alır, oluşturulan grafı (app) çalıştırır ve işlem 
    # bittiğinde sonuçları, üretilen kodu ve varsa hataları düzenli bir AgentResult nesnesi olarak döndürür. 
    # Süreç boyunca oluşan istisnaları (exceptions) yakalar ve loglar, böylece çağıran sisteme (örneğin API 
    # veya UI) her zaman yapılandırılmış ve güvenli bir yanıt dönülmesini garanti eder.
    def process(self, query: str = None) -> AgentResult:

```



Harika, konsepti anladığına sevindim. Dosyalar oldukça kapsamlı olduğu için bunları **iki aşamada** ele alalım.

İlk mesajda; sistemin beyni olan **Router (Yönlendirici)** ve sistemin omurgasını oluşturan **Utility (Yardımcı)** dosyalarını (`base_agent`, `document`, `vectordeneme`) tamamlayalım.

İkinci mesajda ise görselleştirme uzmanı olan `plotgraph.py` dosyasına odaklanabiliriz.

İşte `router_agent.py` ve yardımcı modüller için açıklamalar:

---

### 1. Dosya: `router_agent.py` (Ana Yönlendirici)

Bu dosya sistemin giriş kapısıdır. Hangi ajanın çalışacağına karar verir.

#### `RoutingDecision` (Pydantic Model)

```python
# LLM'in yönlendirme kararı verirken uyması gereken yapılandırılmış veri formatıdır.
# Kullanıcının sorgusunu analiz ettikten sonra hangi uzman ajanın (SQL, Veri Analizi veya API)
# seçildiğini, bu seçimin arkasındaki mantıksal gerekçeyi (reasoning) ve kararın güven
# skorunu (confidence) tutar. Bu yapı, LLM çıktısının kod tarafından işlenebilir olmasını garanti eder.
class RoutingDecision(BaseModel):

```

#### `RouterState` (TypedDict)

```python
# Router grafiğindeki düğümler arasında veri taşımak için kullanılan merkezi durum (state) sözlüğüdür.
# Kullanıcı sorgusu, yönlendirme kararı, seçilen ajanın ürettiği sonuçlar ve varsa hata mesajları
# burada tutulur. Ayrıca, farklı ajanlardan dönen veri çerçevelerini (dataframe) JSON formatında
# saklayarak süreç sonunda birleştirilmelerine veya raporlanmalarına olanak tanır.
class RouterState(TypedDict):

```

#### `__init__`

```python
# LangGraph tabanlı akıllı yönlendiriciyi başlatan kurucu metottur. Yapılandırma dosyasından (config)
# gelen ayarlara göre veritabanı bağlantılarını kurar, CSV dosyalarını yükler ve vektör tabanlı
# bağlam (context) arama motorlarını hazırlar. Ayrıca alt ajanları (Data, SQL, API) başlatarak
# sistemin gelen sorguları işlemeye hazır hale gelmesini sağlar.
def __init__(self, llm: ChatGroq, config: Dict[str, Any]):

```

#### `_get_detailed_resources_summary`

```python
# LLM'in doğru yönlendirme kararı verebilmesi için mevcut kaynakların (Veri setleri, SQL tabloları, API yetenekleri)
# detaylı bir özetini metin formatında oluşturur. CSV dosyasındaki sütun isimlerinden, veritabanındaki
# tablo şemalarına kadar teknik detayları bir araya getirir. Bu özet, yönlendirme prompt'una eklenerek
# yapay zekanın "Elimde hangi veriler var?" sorusuna cevap vermesini sağlar.
def _get_detailed_resources_summary(self) -> str:

```

#### `_llm_routing_node`

```python
# Sistemin beyni olarak görev yapan bu fonksiyon, kullanıcı sorgusunu ve mevcut kaynak özetini
# LLM'e göndererek en uygun ajanın seçilmesini sağlar. Gelen yanıtı 'RoutingDecision' modeline
# göre ayrıştırır (parse eder); eğer geçersiz bir seçim yapılırsa veya hata oluşursa, eldeki
# kaynaklara göre (örneğin CSV varsa Data ajanı) güvenli bir "fallback" (yedek) mekanizması çalıştırır.
def _llm_routing_node(self, state: RouterState) -> RouterState:

```

#### `_data_agent_node`

```python
# Router tarafından "Veri Analizi" (CSV/Pandas) yapılmasına karar verildiğinde çalışan düğümdür.
# `DataAnalysisAgent` sınıfını tetikleyerek analizi başlatır ve dönen sonuçları ana durum (state)
# yapısına işler. Eğer analiz sonucunda bir DataFrame oluştuysa, bunu JSON formatına çevirerek
# state içinde saklar, böylece sonuçlar kaybolmadan kullanıcıya iletilebilir.
def _data_agent_node(self, state: RouterState) -> RouterState:

```

#### `_sql_agent_node`

```python
# Kullanıcı veritabanı ile ilgili bir sorgu sorduğunda (örneğin "Müşterileri listele") devreye giren düğümdür.
# `SQLQuerryAgent` sınıfını kullanarak doğal dili SQL sorgusuna çevirir ve çalıştırır. İşlem sonucunda
# elde edilen verileri ve meta verileri yakalayarak router'ın ana akışına dahil eder. Hata durumunda
# hatayı yakalayıp state'e işleyerek akışın çökmesini engeller.
def _sql_agent_node(self, state: RouterState) -> RouterState:

```

#### `_api_agent_node`

```python
# İçerideki verilerle (CSV/SQL) cevaplanamayacak, dış dünya bilgisi gerektiren (örneğin "Hava durumu", "Güncel haberler")
# sorgular için çalışan düğümdür. `ExternalAPIAgent` aracılığıyla dış kaynaklardan bilgi çeker.
# Bu düğüm, sistemin sadece statik verilerle sınırlı kalmayıp gerçek zamanlı bilgilere de erişebilmesini sağlar.
def _api_agent_node(self, state: RouterState) -> RouterState:

```

#### `_build_graph`

```python
# Router'ın karar mekanizmasını ve iş akışını bir LangGraph grafiği olarak inşa eden fonksiyondur.
# Başlangıç noktası olarak yönlendirme (routing) düğümünü belirler ve LLM'in kararına göre
# akışın hangi alt ajana (Data, SQL veya API) dallanacağını tanımlayan koşullu kenarları (conditional edges) kurar.
def _build_graph(self) -> StateGraph:

```

#### `process`

```python
# Dış dünyadan gelen isteği karşılayan ve tüm router akışını başlatan ana metottur.
# Başlangıç durumunu (initial state) hazırlar, grafiği çalıştırır ve işlem bittiğinde
# elde edilen tüm sonuçları (metin cevapları, dataframe'ler, meta veriler) birleştirerek
# standart bir `AgentResult` nesnesi olarak kullanıcıya döndürür.
def process(self, query: str) -> AgentResult:

```

---

### 2. Dosya: `base_agent.py` (Temel Yapılar)

Bu dosya diğer tüm ajanların miras aldığı veya kullandığı ortak yapıları içerir.

#### `CodeOutput`

```python
# Kod üreten ajanların (Pandas veya SQL) LLM çıktısını standartlaştırmak için kullanılan Pydantic modelidir.
# LLM'in ürettiği ham metnin sadece kod bloğunu içermesini ve belirli bir formatta olmasını zorlar.
# Bu sayede, üretilen kodun "exec" veya SQL motorları tarafından çalıştırılabilir olması kolaylaşır.
class CodeOutput(BaseModel):

```

#### `AgentResult`

```python
# Tüm ajanların (Data, SQL, API, Router) işlem sonucunda döndürdüğü standart sonuç kapsayıcısıdır.
# İşlemin başarılı olup olmadığını (success), elde edilen veriyi (data), varsa hata mesajını (error)
# ve ek bilgileri (metadata) tek bir obje içinde tutarak sistem genelinde tutarlı bir iletişim sağlar.
class AgentResult:

```

#### `BaseSpecializedAgent`

```python
# Tüm özelleşmiş ajanların (DataAnalysisAgent, SQLQuerryAgent vb.) türetildiği soyut temel sınıftır (Abstract Base Class).
# Ajanların sahip olması gereken temel özellikleri (LLM bağlantısı, loglama yetenekleri) ve mutlaka
# uygulamaları gereken `process` metodunu tanımlayarak kod standardizasyonu sağlar.
class BaseSpecializedAgent(ABC):

```

#### `is_safe_code`

```python
# Üretilen Python kodunu çalıştırmadan önce güvenlik taramasından geçiren kritik bir fonksiyondur.
# Kod içerisinde sistem dosyalarına erişim (os, sys), dosya silme veya dış komut çalıştırma gibi
# tehlikeli anahtar kelimelerin olup olmadığını kontrol eder. Güvenli olmayan kodların çalışmasını engelleyerek sistemi korur.
def is_safe_code(code: str) -> bool:

```

---

### 3. Dosya: `document.py` (Döküman İşleme)

#### `DocumentProcessor`

```python
# Farklı formatlardaki (PDF, TXT, DOCX) dökümanları okuyup ham metne dönüştüren yardımcı sınıftır.
# Dosya uzantısına göre uygun ayrıştırıcıyı (parser) seçer ve metni temizleyerek çıkarır.
# Özellikle RAG (Retrieval-Augmented Generation) süreçlerinde veritabanı şemaları veya iş kuralları dökümanlarını sisteme beslemek için kullanılır.
class DocumentProcessor:

```

---

### 4. Dosya: `vectordeneme.py` (Vektör Veritabanı / RAG)

#### `VectorStore`

```python
# ChromaDB kullanarak metin verilerinin vektör (sayısal temsil) formatında saklanmasını ve yönetilmesini sağlayan sınıftır.
# Belgeleri embedding (gömme) modelleriyle vektörlere dönüştürür, veritabanına kaydeder ve
# anlamsal benzerlik araması (semantic search) yaparak sorguya en yakın içeriklerin bulunmasını sağlar.
class VectorStore:

```

#### `ContextFind`

```python
# RAG (Retrieval-Augmented Generation) yapısının temelini oluşturan sınıftır. Verilen bir dökümanı
# parçalara böler (chunking), vektör veritabanına yükler ve kullanıcı bir soru sorduğunda
# döküman içinden en alakalı kısımları (context) bulup getirir. Bu bağlam, LLM'in sorulara dökümana dayalı cevap vermesini sağlar.
class ContextFind:

```

---

Harika, şimdi sistemin görselleştirme kanadını yöneten **`plotgraph.py`** dosyasını ele alalım. Bu ajan, veri analizi ajanına benzer bir yapıdadır ancak çıktısı veri değil, görsel grafiklerdir ve kendine özgü doğrulama (validation) süreçleri içerir.

İşte `plotgraph.py` için açıklamalar:

### `VisualizationErrorCorrection`

```python
# Görselleştirme sürecinde bir hata oluştuğunda (örneğin; yanlış kütüphane kullanımı veya veri tipi uyuşmazlığı) 
# LLM'den beklenen yapılandırılmış düzeltme formatıdır. Hatanın analizini, bir sonraki adımda hangi düğüme 
# (node) gidilmesi gerektiğini ve hatayı gidermek için yeniden yazılmış sorguyu içerir. Bu yapı sayesinde, 
# sistem körü körüne tekrar denemek yerine, hatanın nedenini anlayarak stratejik bir düzeltme uygular.
class VisualizationErrorCorrection(BaseModel):

```

### `VisualizationValidationResult`

```python
# Üretilen görselleştirme kodunun (henüz çalıştırılmadan veya çalıştırıldıktan sonra) mantıksal olarak 
# kullanıcının isteğini karşılayıp karşılamadığını değerlendiren modeldir. Görseli doğrudan "göremese" bile, 
# kodun kullandığı sütunları, grafik türünü ve mantığını analiz ederek bir doğruluk skoru ve geri bildirim üretir. 
# Bu geri bildirim, grafiğin eksik veya yanlış olması durumunda kendi kendini düzeltme döngüsünü tetikler.
class VisualizationValidationResult(BaseModel):

```

### `PlotExecutionResult`

```python
# Grafik oluşturma işleminin sonucunu taşıyan kapsayıcı sınıftır. Kodun başarıyla çalışıp çalışmadığını, 
# üretilen Python kodunu ve en önemlisi oluşturulan resim dosyasının (PNG/JPG) diskteki yolunu (path) tutar. 
# Bu yapı, oluşturulan görselin dosya sisteminden alınıp kullanıcı arayüzüne veya rapora taşınmasını sağlayan köprüdür.
class PlotExecutionResult(BaseModel):

```

### `VisualizationState`

```python
# Görselleştirme grafiğindeki (graph) tüm düğümler arasında paylaşılan merkezi hafızadır. Kullanıcı sorgusunu, 
# üretilen Matplotlib/Seaborn kodlarını, dosya yollarını, doğrulama sonuçlarını ve hata durumlarını saklar. 
# Ajanın "o an ne yaptığını" ve "geçmişte ne denediğini" bilmesini sağlayarak çok adımlı (multi-step) işlemleri yönetir.
class VisualizationState(TypedDict):

```

### `VisualizationAgent` (`__init__`)

```python
# Görselleştirme ajanını başlatan, gerekli klasörleri (plots/) oluşturan ve görselleştirme kütüphanelerinin 
# (matplotlib, seaborn) ayarlarını yapan kurucu metottur. Ayrıca, veri çerçevesini (DataFrame) ve varsa 
# dökümanları yükleyerek analize hazır hale getirir. Grafiklerin kaydedileceği dizini garanti altına alır 
# ve görsel stil ayarlarını (style, palette) yaparak çıktıların estetik standardını belirler.
def __init__(self, llm: ChatGroq, df: pd.DataFrame, doc_path: str = None, column_info_path: str = None, plots_dir: str = "plots"):

```

### `_llm_routing_node`

```python
# Kullanıcıdan gelen görselleştirme isteğini analiz ederek en uygun iş akışını belirleyen karar mekanizmasıdır. 
# İsteğin "sadece kod üretimi", "tam görselleştirme" veya "veri bilgisi alma" olup olmadığına karar verir. 
# Bu yönlendirme, gereksiz grafik çizimlerini engeller veya kullanıcının sadece kod istediği senaryolara 
# uyum sağlayarak işlem maliyetini düşürür.
def _llm_routing_node(self, state: VisualizationState) -> VisualizationState:

```

### `get_data_info`

```python
# Veri setindeki sütunların tiplerini (sayısal vs kategorik) ayrıştırarak LLM'e görselleştirme için 
# optimize edilmiş bir özet sunar. Hangi sütunların eksenlerde kullanılabileceği, hangilerinin gruplama 
# (hue) için uygun olduğu gibi ipuçlarını hazırlar. Bu adım, LLM'in "String bir sütunun ortalamasını almaya çalışmak" 
# gibi temel hatalar yapmasını engellemek için kritiktir.
def get_data_info(self, state: VisualizationState) -> VisualizationState:

```

### `generate_plot_code`

```python
# Toplanan veri özetini ve kullanıcı isteğini alarak çalıştırılabilir Matplotlib veya Seaborn kodu üretir. 
# LLM'e özel talimatlar vererek kodun `plt.figure` ile başlamasını, `plt.savefig` ile bitmesini ve 
# estetik açıdan düzgün olmasını (başlıklar, etiketler vb.) zorlar. Üretilen kodun güvenli, temiz 
# ve doğrudan çalıştırılabilir formatta olmasını sağlar.
def generate_plot_code(self, state: VisualizationState) -> VisualizationState:

```

### `create_visualization`

```python
# Üretilen grafik kodunu izole bir ortamda çalıştırarak (exec) görsel dosyasını diskte oluşturur. 
# Kod çalışırken oluşabilecek hataları (RuntimeError vb.) yakalar ve kodun dosya sistemine gerçekten 
# bir resim kaydedip kaydetmediğini kontrol eder. Başarılı olursa, oluşturulan dosyanın yolunu 
# (path) sonuç olarak döndürür; aksi takdirde hata yönetimi sürecini başlatır.
def create_visualization(self, state: VisualizationState) -> VisualizationState:

```

### `validate_result`

```python
# Oluşturulan kodun mantıksal doğruluğunu denetleyen "eleştirmen" fonksiyonudur. LLM, grafiği göremese de, 
# kodun kullanıcının isteğine uygun sütunları kullanıp kullanmadığını ve doğru grafik türünü (bar, scatter vb.) 
# seçip seçmediğini kod üzerinden analiz eder. Eğer kod mantıksal olarak hatalıysa (örneğin zaman serisi için bar chart kullanımı), 
# yeniden üretim için geri bildirim oluşturur.
def validate_result(self, state: VisualizationState) -> VisualizationState:

```

### `enhance_query_with_feedback`

```python
# Doğrulama aşamasında tespit edilen eksiklikleri gidermek için orijinal sorguyu teknik notlarla zenginleştirir. 
# Örneğin, validasyon sonucu "Eksik veri temizlenmemiş" derse, bu fonksiyon sorguya "Lütfen önce eksik verileri 
# temizle (dropna) ve sonra çiz" gibi bir talimat ekler. Bu sayede bir sonraki kod üretim denemesi çok daha isabetli olur.
def enhance_query_with_feedback(self, state: VisualizationState) -> VisualizationState:

```

### `error_handling` ve `_llm_error_correction`

```python
# Görselleştirme sürecindeki hataları (Syntax hatası, boyut uyuşmazlığı, yanlış sütun ismi) yöneten ve düzelten modüldür. 
# Hatayı sadece raporlamak yerine, hatanın nedenini (Root Cause Analysis) LLM ile tespit eder ve düzeltilmiş bir 
# talimat seti oluşturur. Sistem belirli bir tekrar sayısına (max_retries) ulaşana kadar kodu düzeltip yeniden çizmeyi dener.
def error_handling(self, state: VisualizationState) -> VisualizationState:

```

### `_build_graph`

```python
# Tüm bu görselleştirme adımlarını (Bilgi Al -> Kod Üret -> Çiz -> Doğrula -> Hata Düzelt) birbirine bağlayan 
# akış diyagramını (LangGraph) oluşturur. Hangi durumda hangi adıma geçileceğini, döngülerin nerede başlayıp 
# biteceğini tanımlar. Bu yapı, lineer olmayan, hatalardan dönebilen esnek bir görselleştirme süreci sağlar.
def _build_graph(self) -> StateGraph:

```
