#######################################
# AB Test Project
#######################################

# İş Problemi:
# Facebook kısa süre önce mevcut maximum bidding adı verilen teklif
# verme türüne alternatif olarak yeni bir teklif türü olan average bidding’i
# tanıttı.

# Müşterilerimizden biri olan bombabomba.com, bu yeni özelliği test
# etmeye karar verdi ve averagebidding’in, maximumbidding’den daha
# fazla dönüşüm getirip getirmediğini anlamak için bir A/B testi yapmak
# istiyor.

#########################################
# Veri Seti Hikayesi
#########################################

# bombabomba.com’un web site bilgilerini içeren bu veri setinde kullanıcıların
# gördükleri ve tıkladıkları reklam sayıları gibi bilgilerin yanı sıra buradan gelen
# kazanç bilgileri yer almaktadır.

# Kontrol ve Test grubu olmak üzere iki ayrı veri seti vardır.

##########################################
# Değişkenler
##########################################

# Impression – Reklam görüntüleme sayısı
# Click – Tıklama
## Görüntülenen reklama tıklanma sayısını belirtir.
# Purchase – Satın alım
## Tıklanan reklamlar sonrası satın alınan ürün sayısını belirtir.
# Earning – Kazanç
## Satın alınan ürünler sonrası elde edilen kazanç

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('isplay.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


##########################################
# Proje Görevleri
##########################################

# Average teklif verme Maximum teklif vermekten daha mı fazla dönüşüm getirir?

# Görev 1
# A/B testinin hipotezini tanımlayınız.

control_df = pd.read_excel("weeks/week_05/ab_testing.xlsx", sheet_name= "Control Group" )
control_df.head()
control_df.shape()
control_df.describe().T


test_df = pd.read_excel("weeks/week_05/ab_testing.xlsx", sheet_name= "Test Group")

test_df.head()
test_df.shape()
test_df.describe().T

control_df["Purchase"].mean() # 550.8940587702316

test_df["Purchase"].mean() # 582.1060966484675

#test grubunun ortalaması control grubunun ortalamasından yüksek.


# Confidence Intervals (Güven Aralıkları)

control_df.describe().T

import statsmodels.stats.api as sms

sms.DescrStatsW(control_df["Purchase"]).tconfint_mean() # (508.0041754264924, 593.7839421139709) kontrol grubu %95 güven aralığı değerleri.

sms.DescrStatsW(test_df["Purchase"]).tconfint_mean() # (530.5670226990063, 633.645170597929)

# Hypothesis Testing (Hipotez Testi)

# AB Testing (Bağımsız İki Örneklem T Testi)

# İki grup ortalaması arasında karşılaştırma yapılmak istenildiğinde kullanılır.

# 1. Varsayım Kontrolü
#   - 1. Normallik Varsayımı
#   - 2. Varyans Homojenliği
# 2. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# Not:
# - Normallik sağlanmıyorsa direk 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.


# Tıklanan reklamlar sonrası satın alınan ürün sayısına'a göre Tıklanma ve Reklam Görüntüleme ortalamaları
control_df.groupby("Purchase").agg({"Click": "mean", "Impression" : "mean"})

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Norma dağılım varsayımı sağlanmamaktadır.
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal

test_stat, pvalue = shapiro(control_df["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) # Test Stat = 0.9773, p-value = 0.5891

# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

# p-value değerimiz 0.05den büyük çıktı. o yüzden H0 hipotezi reddedilemez. control_df değerleri normal dağılmıştır denebilir.

test_stat, pvalue = shapiro(test_df["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = 0.9589, p-value = 0.1541

# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

# p-value değerimiz 0.05den büyük çıktı. o yüzden H0 hipotezi reddedilemez. test_df değerleri normal dağılmıştır denebilir.


############################
# Varyans Homojenligi Varsayımı
############################

# VARYANS HOMOJENLİĞİ KONTROLÜ

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(control_df["Purchase"], test_df["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = 2.6393, p-value = 0.1083

# # p_value değeri 0.05'ten büyük olduğu için H0 hipotezini reddedemeyiz.


############################
# Hipotezin Uygulanması
############################

# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)

# Eğer normallik sağlanmazsa her türlü nonparametrik test yapacağız.
# Eger normallik sağlanır varyans homojenliği sağlanmazsa ne olacak?
# T test fonksiyonuna arguman gireceğiz.


# H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)


############################
# Varsayımlar sağlandığı için bağımsız iki örneklem t testi (parametrik test)
############################

test_stat, pvalue =  ttest_ind(control_df["Purchase"], test_df["Purchase"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = -0.9416, p-value = 0.3493 p-value > 0.05den yüksek o yüzden arasında istatiksel olarak bir farklılık yoktur denebilir.


# İKİ ÖRNEKLEM ORAN TESTİ
    # H0 : Kontrol grubu purchase dönüşüm oranı ile test grubu purchase dönüşüm oranı arasında istatistiksel olarak anlamlı bir fark yoktur.
    # H1 : Kontrol grubu purchase dönüşüm oranı ile test grubu purchase dönüşüm oranı arasında istatistiksel olarak anlamlı bir fark vardır.

control_df["Purchase"].shape[0]     # Gözlem sayısı 40, n1>30 varsayımı sağlanıyor.
test_df["Purchase"].shape[0]        # Gözlem sayısı 40, n2>30 varsayımı sağlanıyor.

control_df["Purchase"].sum()
control_df["Impression"].sum()

test_df["Purchase"].sum()
test_df["Impression"].sum()

from statsmodels.stats.proportion import proportions_ztest
basari_sayisi = np.array([control_df["Purchase"].sum(), test_df["Purchase"].sum()])
gozlem_sayisi = np.array([control_df["Impression"].sum(), test_df["Impression"].sum()])

ttest_z, p_value_z = proportions_ztest(basari_sayisi, gozlem_sayisi)
print("ttest istatistiği: {}\np_value: {:.10f}".format(ttest_z, p_value_z))
# p_value değeri 0.05'ten küçük olduğu için H0 hipotezi reddedilir. Yani iki grubun purchase dönüşüm oranları arasında anlamlı bir farklılık vardır.



control_df["Purchase"].sum()/control_df["Impression"].sum() #  0.00541624432470298
test_df["Purchase"].sum()/test_df["Impression"].sum() # 0.004830258461839089


control_df["Purchase"].sum()/control_df["Click"].sum() # 0.10800452148227198
test_df["Purchase"].sum()/test_df["Click"].sum() # 0.14671677275453024


# Oranlar birbirine zıt çıkmıştır. Hipotezlerden sonuç alabilmek için takibe devam edilmeli.
