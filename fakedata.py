import random
import psycopg2
from faker import Faker

fake = Faker()

# PostgreSQL bağlantı bilgileri
conn = psycopg2.connect(
    dbname="musteri_db",
    user="postgres",
    password="123",
    host="localhost",
    port="5432"
)

cur = conn.cursor()

# Kaç sahte kayıt istiyorsan
for _ in range(100):
    values = (
        random.choice([True, False]),  # exist_retail_credit_card
        random.choice([True, False]),  # exist_black_card
        random.choice([True, False]),  # exist_seller_card
        random.choice([True, False]),  # exist_proemtia_seller_card
        random.choice([True, False]),  # exist_exporter_card
        random.choice([True, False]),  # exist_tim_exporter_card
        random.choice([True, False]),  # exist_cks
        random.choice([True, False]),  # exist_tardes
        random.choice([True, False]),  # exist_tmo_commitment_iscep
        random.choice([True, False]),  # exist_farmer_ktmh_risk
        random.choice([True, False]),  # exist_imece_member_business
        random.choice([True, False]),  # is_abom
        round(random.uniform(0, 100000), 2),  # exist_krs_ank_cash_risk
        round(random.uniform(0, 50000), 2),   # exist_isb_cash_limit
        round(random.uniform(0, 50000), 2),   # exist_isb_non_cash_risk
        round(random.uniform(0, 150000), 2),  # exist_isb_total_risk
        round(random.uniform(0, 70000), 2),   # exist_krs_bnk_noncash_limit
        round(random.uniform(0, 100000), 2),  # exist_krs_total_noncash_limit
        round(random.uniform(0, 70000), 2),   # exist_krs_bnk_noncash_risk
        round(random.uniform(0, 150000), 2),  # exist_krs_total_total_limit
        round(random.uniform(0, 150000), 2),  # exist_krs_total_total_risk
        round(random.uniform(0, 50000), 2),   # exist_krs_bnk_cash_limit
        round(random.uniform(0, 100000), 2),  # exist_krs_total_cash_limit
        round(random.uniform(0, 80000), 2),   # exist_krs_total_cash_risk
        round(random.uniform(0, 100000), 2),  # exist_krs_total_noncash_risk
        fake.word(),  # personality_type
        random.choice([True, False]),  # commercial_credit_cart
        random.choice([True, False]),  # installment_loan
        fake.word(),  # qualification
        fake.word(),  # sector
        fake.word(),  # risk_group
        fake.word(),  # segment_nane
        random.choice([True, False]),  # customer_activity
        fake.lexify(text='?????'),     # curr_total_score_letter
        random.choice([True, False]),  # possibility_helt_ing
        random.choice([True, False]),  # foreign_trade_customer
        fake.word(),                   # ltv_value_grade
        random.choice([True, False]),  # youth_kobi
        random.choice([True, False]),  # customer_loyalty
        random.choice([True, False]),  # female_entrepreneur
        random.choice([True, False]),  # exporter_associations
        random.choice([True, False]),  # ktni
        random.choice([True, False]),  # instalment_kt
        random.choice([True, False]),  # invoice_discount
        random.choice([True, False]),  # dos
        random.choice([True, False]),  # guarantee_letter
        random.choice([True, False]),  # digital_supplier_finance
        random.choice([True, False]),  # instant_trade_credit
        random.choice([True, False]),  # digital_bch
        random.choice([True, False]),  # cencepte
        random.choice([True, False]),  # iscep_individual
        random.choice([True, False]),  # commercial_internet_branch
        random.choice([True, False]),  # maximum_business_place
        random.choice([True, False]),  # maximum_mobile
        random.choice([True, False]),  # individual_internet_branch
        random.choice([True, False]),  # iscep_commercial
        fake.word(),                   # evaluation_type
        random.choice([True, False]),  # branch_authorized
        random.choice([True, False]),  # exist_isr_srk_limit_bosluk
        fake.word(),                   # exportation
        fake.word(),                   # integrated_score_result
        random.choice([True, False]),  # package_tariff
        random.choice([True, False])   # cek_karnesi
    )

    cur.execute("""
        INSERT INTO musteri_verisi (
            exist_retail_credit_card, exist_black_card, exist_seller_card,
            exist_proemtia_seller_card, exist_exporter_card, exist_tim_exporter_card,
            exist_cks, exist_tardes, exist_tmo_commitment_iscep,
            exist_farmer_ktmh_risk, exist_imece_member_business, is_abom,
            exist_krs_ank_cash_risk, exist_isb_cash_limit, exist_isb_non_cash_risk,
            exist_isb_total_risk, exist_krs_bnk_noncash_limit, exist_krs_total_noncash_limit,
            exist_krs_bnk_noncash_risk, exist_krs_total_total_limit, exist_krs_total_total_risk,
            exist_krs_bnk_cash_limit, exist_krs_total_cash_limit, exist_krs_total_cash_risk,
            exist_krs_total_noncash_risk, personality_type, commercial_credit_cart,
            installment_loan, qualification, sector, risk_group, segment_nane,
            customer_activity, curr_total_score_letter, possibility_helt_ing,
            foreign_trade_customer, ltv_value_grade, youth_kobi, customer_loyalty,
            female_entrepreneur, exporter_associations, ktni, instalment_kt,
            invoice_discount, dos, guarantee_letter, digital_supplier_finance,
            instant_trade_credit, digital_bch, cencepte, iscep_individual,
            commercial_internet_branch, maximum_business_place, maximum_mobile,
            individual_internet_branch, iscep_commercial, evaluation_type,
            branch_authorized, exist_isr_srk_limit_bosluk, exportation,
            integrated_score_result, package_tariff, cek_karnesi
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, values)

conn.commit()
cur.close()
conn.close()
