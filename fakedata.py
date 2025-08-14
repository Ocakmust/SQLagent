import random
import psycopg2
from faker import Faker

fake = Faker()

# PostgreSQL bağlantısı
conn = psycopg2.connect(
    dbname="musteri_db",
    user="postgres",
    password="123",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# Tabloyu sil ve yeniden oluştur
cur.execute("DROP TABLE IF EXISTS musteri_verisi")

cur.execute("""
CREATE TABLE musteri_verisi (
    exist_retail_credit_card BOOLEAN,
    commercial_credit_cart BOOLEAN,
    exist_158_skk_risk_ol BOOLEAN,
    exist_isr_srk_limit_bosluk BOOLEAN,
    exist_seller_card BOOLEAN,
    exist_proemtia_seller_card BOOLEAN,
    exist_exporter_card BOOLEAN,
    exist_tim_exporter_card BOOLEAN,
    exporter_associations BOOLEAN,
    exist_cks BOOLEAN,
    exist_tardes BOOLEAN,
    exist_tmo_commitment_iscep BOOLEAN,
    exist_farmer_ktmh_risk BOOLEAN,
    exist_imece_member_business BOOLEAN,
    exist_krs_ank_cash_risk NUMERIC,
    exist_krs_bnk_cash_limit NUMERIC,
    exist_krs_total_cash_limit NUMERIC,
    exist_krs_total_cash_risk NUMERIC,
    exist_krs_bnk_noncash_risk NUMERIC,
    exist_krs_total_noncash_limit NUMERIC,
    exist_krs_total_noncash_risk NUMERIC,
    exist_krs_total_total_limit NUMERIC,
    exist_krs_total_total_risk NUMERIC,
    exist_isb_cash_limit NUMERIC,
    exist_isb_non_cash_risk NUMERIC,
    exist_isb_total_risk NUMERIC,
    is_abom BOOLEAN,
    customer_activity BOOLEAN,
    possibility_helting BOOLEAN,
    youth_kobi BOOLEAN,
    customer_loyalty BOOLEAN,
    female_entrepreneur BOOLEAN,
    commercial_internet_branch BOOLEAN,
    individual_internet_branch BOOLEAN
  
)
""")

# Veri üretimi
for _ in range(100):
    values = (
        random.choice([True, False]),  # exist_retail_credit_card
        random.choice([True, False]),  # commercial_credit_cart
        random.choice([True, False]),  # exist_158_skk_risk_ol
        random.choice([True, False]),  # exist_isr_srk_limit_bosluk
        random.choice([True, False]),  # exist_seller_card
        random.choice([True, False]),  # exist_proemtia_seller_card
        random.choice([True, False]),  # exist_exporter_card
        random.choice([True, False]),  # exist_tim_exporter_card
        random.choice([True, False]),  # exporter_associations
        random.choice([True, False]),  # exist_cks
        random.choice([True, False]),  # exist_tardes
        random.choice([True, False]),  # exist_tmo_commitment_iscep
        random.choice([True, False]),  # exist_farmer_ktmh_risk
        random.choice([True, False]),  # exist_imece_member_business
        round(random.uniform(0, 100000), 2),  # exist_krs_ank_cash_risk
        round(random.uniform(0, 100000), 2),  # exist_krs_bnk_cash_limit
        round(random.uniform(0, 150000), 2),  # exist_krs_total_cash_limit
        round(random.uniform(0, 150000), 2),  # exist_krs_total_cash_risk
        round(random.uniform(0, 80000), 2),   # exist_krs_bnk_noncash_risk
        round(random.uniform(0, 150000), 2),  # exist_krs_total_noncash_limit
        round(random.uniform(0, 150000), 2),  # exist_krs_total_noncash_risk
        round(random.uniform(0, 200000), 2),  # exist_krs_total_total_limit
        round(random.uniform(0, 200000), 2),  # exist_krs_total_total_risk
        round(random.uniform(0, 50000), 2),   # exist_isb_cash_limit
        round(random.uniform(0, 50000), 2),   # exist_isb_non_cash_risk
        round(random.uniform(0, 150000), 2),  # exist_isb_total_risk
        random.choice([True, False]),  # is_abom
        random.choice([True, False]),  # customer_activity
        random.choice([True, False]),  # possibility_helting
        random.choice([True, False]),  # youth_kobi
        random.choice([True, False]),  # customer_loyalty
        random.choice([True, False]),  # female_entrepreneur
        random.choice([True, False]),  # commercial_internet_branch
        random.choice([True, False])  # individual_internet_branch
    )

    cur.execute("""
        INSERT INTO musteri_verisi (
            exist_retail_credit_card, commercial_credit_cart,
            exist_158_skk_risk_ol, exist_isr_srk_limit_bosluk,
            exist_seller_card, exist_proemtia_seller_card,
            exist_exporter_card, exist_tim_exporter_card,
            exporter_associations, exist_cks, exist_tardes,
            exist_tmo_commitment_iscep, exist_farmer_ktmh_risk,
            exist_imece_member_business, exist_krs_ank_cash_risk,
            exist_krs_bnk_cash_limit, exist_krs_total_cash_limit,
            exist_krs_total_cash_risk, exist_krs_bnk_noncash_risk,
             exist_krs_total_noncash_limit,
            exist_krs_total_noncash_risk, exist_krs_total_total_limit,
            exist_krs_total_total_risk, exist_isb_cash_limit,
            exist_isb_non_cash_risk, exist_isb_total_risk, is_abom,
            customer_activity, possibility_helting, youth_kobi,
            customer_loyalty, female_entrepreneur,
            commercial_internet_branch, individual_internet_branch
        )
        VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, values)

conn.commit()
cur.close()
conn.close()
