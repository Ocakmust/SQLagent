"""
•
EXIST_RETAIL_CREDIT_CARD Müşterinin bireysel kredi kartına sahip olup olmadığını belirten bir alan. Bu değer, müşterinin bireysel bankacılık kapsamında kredi kartı ürünü kullanıp kullanmadığını gösterir. Örneğin: "Müşterinin Akbank bireysel kredi kartı var mı?" gibi sorular bu alanla yanıtlanabilir.
•
COMMERCIAL_CREDIT_CART Müşterinin ticari bir kredi kartına sahip olup olmadığını gösteren alandır. Bu, şirket adına açılmış ve firma harcamaları için kullanılan kartları kapsar. Ticari müşteri ayrımı veya harcama analizleri için bu alan kullanılabilir.
•
EXIST_158_SKK_RISK_OL Müşterinin ticari kredi kartına ait bir limiti olup olmadığını ve bu limit üzerinde aktif bir risk (yani borç, bakiye) bulunup bulunmadığını belirtir. Hem kredi kartı limiti tanımlanmış hem de kullanılmış olan müşterileri bulmak için kullanılır.
•
EXIST_ISR_SRK_LIMIT_BOSLUK Müşterinin ticari kredi kartına ait tanımlanmış bir limitinin olup olmadığını ve bu limitte henüz kullanılmamış (boşta) bir bakiye bulunup bulunmadığını gösterir. Risk analizi veya limit güncelleme önerileri için kullanılabilir.
•
EXIST_SELLER_CARD Müşterinin bayi (satıcı) kartı sahibi olup olmadığını gösterir. Bayi kartı, genellikle ticari ilişkilerde kullanılır. Şirketin tedarikçi veya bayilere sunduğu özel finansman kartı türlerini içerir.
•
EXIST_PROEMTIA_SELLER_CARD Müşterinin “Proemtia” markalı özel bir bayi kartına sahip olup olmadığını gösterir. Bu tür kartlar spesifik iş ortaklıkları veya iş modellerine özgü ürünler olabilir.
•
EXIST_EXPORTER_CARD Müşterinin “Maximum İhracatçı Kartı” gibi dış ticaret işlemlerine özel bir ürünü olup olmadığını belirtir. Bu tür kartlar ihracatçılara yönelik özel kampanyalar, döviz işlemleri ve gümrük avantajları sunabilir.
•
EXIST_TIM_EXPORTER_CARD Müşterinin TİM (Türkiye İhracatçılar Meclisi) tarafından tanımlanmış bir ihracatçı kartına sahip olup olmadığını gösterir. Bu tür kartlar, kurumsal ihracatçıları ayırt etmek ve desteklemek amacıyla kullanılır.
•
EXPORTER_ASSOCIATIONS Müşterinin herhangi bir ihracatçı birliğine (örneğin Ege İhracatçı Birlikleri) üye olup olmadığını gösterir. Kurumsal kimlik ve ihracat statüsü değerlendirmelerinde önemlidir.
•
EXIST_CKS Müşterinin ÇKS (Çiftçi Kayıt Sistemi) kaydının bulunup bulunmadığını belirtir. Bu alan, müşterinin çiftçi statüsünde olup olmadığını anlamak için kullanılır. Tarım destek kredileri veya devlet teşvikleri açısından kritiktir.
•
EXIST_TARDES Müşterinin TARDES sistemine kayıtlı olup olmadığını belirtir. TARDES, tarımsal destek ödemelerini takip eden dijital bir sistemdir. Bu bilgi, tarım bankacılığı ürünleri önerileri için kullanılabilir.
•
EXIST_TMO_COMMITMENT_ISCEP Müşterinin İşCep mobil uygulaması üzerinden TMO (Toprak Mahsulleri Ofisi) taahhütnamesi verip vermediğini gösterir. Tarım ürünü satış taahhütleri ile ilgilidir ve TMO ile çalışan üreticilerin tespiti için kullanılır.
•
EXIST_FARMER_KTMH_RISK Müşterinin çiftçi ya da tarım sektörüne yönelik kullanılan KTMH (Kredi Tahsis Müşteri Hakkı) kapsamında riskli (borçlu) olup olmadığını belirtir. Tarım kredisi risk raporlamalarında kullanılır.
•
EXIST_IMECE_MEMBER_BUSINESS Müşterinin İmece POS sistemine (tarımsal satışlar için dijital ödeme çözümü) kayıtlı olup olmadığını belirtir. Bu bilgi, POS hizmeti kullanan tarım işletmelerinin ayıklanması için kullanılabilir.
•
EXIST_KRS_ANK_CASH_RISK Müşterinin Kredi Risk Sistemi (KRS) üzerinde bankamıza ait nakdi riski (kullandığı krediler) olup olmadığını gösterir. Bu alan, banka tarafından müşteriye sağlanan kredi borçlarının varlığını belirtir.
•
EXIST_KRS_BNK_CASH_LIMIT Müşterinin bankamız nezdinde tanımlı KRS sistemine kayıtlı nakdi kredi limiti bulunup bulunmadığını belirtir. Henüz kullanılıp kullanılmadığı fark etmeksizin tanımlı limiti ifade eder.
•
EXIST_KRS_TOTAL_CASH_LIMIT Müşterinin tüm bankalar nezdindeki toplam nakdi kredi limitini belirtir. Müşterinin piyasadaki genel kredi erişimi hakkında fikir verir.
•
EXIST_KRS_TOTAL_CASH_RISK Müşterinin tüm bankalara ait toplam nakdi riskini gösterir. Bu, bireyin ya da kurumun finansal borçluluğunu anlamak için kritik bir göstergedir.
•
EXIST_KRS_BNK_NONCASH_RISK Müşterinin bankamız üzerinden yürüttüğü gayri nakdi risklerin (örneğin teminat mektubu, kefalet vb.) varlığını belirtir.
•
EXIST_KRS_TOTAL_NONCASH_LIMIT Tüm finans kuruluşları tarafından müşteriye tanımlanmış olan gayri nakdi limitlerin toplamıdır. Söz konusu müşteri için potansiyel teminat ve kefalet kullanımlarını ölçmekte kullanılır.
•
EXIST_KRS_TOTAL_NONCASH_RISK Müşterinin toplam gayri nakdi risklerini (tüm bankalardaki teminat mektubu borçları gibi) gösterir. Finansal taahhütler değerlendirilirken bu alana bakılır.
•
EXIST_KRS_TOTAL_TOTAL_LIMIT Müşterinin toplam kredi limiti (nakdi + gayri nakdi) bilgisi. Kredi onay ve limit tahsis süreçlerinde kullanılır.
•
EXIST_KRS_TOTAL_TOTAL_RISK Müşterinin toplam kredi riski (nakdi + gayri nakdi) bilgisidir. Bu değer, müşterinin mevcut finansal yükünü net olarak verir.
•
EXIST_ISB_CASH_LIMIT Müşteriye bankamızca tahsis edilmiş nakdi kredi limiti olup olmadığını gösterir. Örnek: "Bu müşteri aktif bir kredi limiti sahibi mi?"
•
EXIST_ISB_NON_CASH_RISK Bankamız nezdinde, müşterinin gayri nakdi kredi ürünleri kaynaklı aktif risklerinin (örneğin teminat mektubu yükümlülükleri) bulunup bulunmadığını gösterir.
•
EXIST_ISB_TOTAL_RISK Müşterinin bankamızdaki toplam riskini (hem nakdi hem gayri nakdi) ifade eder. Tüm borç ve yükümlülüklerin toplamı olarak düşünülebilir.
•
IS_ABOM Müşterinin ABOM (Alternatif Bankacılık Organizasyon Modeli) kapsamında özel bir segmentte tanımlı olup olmadığını belirtir. Bu müşteriler genellikle özel ilişki yöneticisi tarafından yönetilir.
•
CUSTOMER_ACTIVITY Müşterinin aktif bir kullanıcı olup olmadığını belirtir. Aktiflik; son dönemde yapılan işlemler, ürün kullanımı, giriş sıklığı gibi etkenlere bağlıdır.
•
POSSIBILITY_HELTING Müşterinin "erime" yani bankadan ayrılma, pasifleşme, ürünleri kapatma gibi eğilimlerinin olup olmadığını belirtir. Erime olasılığı tahmine dayalı analitik modellerle belirlenir.
•
YOUTH_KOBI Müşterinin genç girişimci veya genç KOBİ (Küçük ve Orta Ölçekli İşletme) sınıfında olup olmadığını gösterir. Yaş, faaliyet süresi, ciro gibi kriterlerle belirlenebilir.
•
CUSTOMER_LOYALTY Müşterinin bankaya olan bağlılık düzeyini ifade eder. Sadakat seviyesi yüksek olan müşteriler genellikle uzun süreli, çok ürünlü ve düzenli kullanım sağlayan bireylerdir.
•
FEMALE_ENTREPRENEUR Müşterinin kadın girişimci statüsünde olup olmadığını belirtir. Cinsiyet, faaliyet türü ve destek programlarına katılım kriterlerine göre sınıflanabilir.
•
COMMERCIAL_INTERNET_BRANCH Müşteri, ticari internet şubesi kanalı üzerinden işlem yapıyor mu? Bu bilgi ticari dijital müşteri davranışlarını analiz etmek için kullanılır.
•
INDIVIDUAL_INTERNET_BRANCH Müşteri, bireysel internet bankacılığına giriş yapmış mı? Web üzerinden işlem yapma eğilimleri ölçülür."""