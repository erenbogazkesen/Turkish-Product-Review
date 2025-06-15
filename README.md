# 🇹🇷 Türkçe Ürün Yorumlarıyla Duygu Analizi + XAI

Bu projede, Türkçe ürün yorumları kullanılarak BERT tabanlı bir duygu analizi modeli eğitilmiş ve model kararları XAI yöntemleriyle açıklanmıştır.

---

## 🚀 Kullanılan Model

- Model: `dbmdz/bert-base-turkish-uncased`
- Eğitim Seti: 10.000 ürün yorumu (5k pozitif, 5k negatif)
- Accuracy: **%88.7**
- Precision: **%85.3**
- Recall: **%93.7**
- F1: **%89.3**

---

## 🎯 Model Performansı

![g ](https://github.com/erenbogazkesen/Turkish-Product-Review/blob/main/images/Performans%20Metrikleri.png?raw=true)

---

## 🧠 XAI Açıklamaları

### 1. Cümle LIME Örneği

![LIME Açıklaması](https://github.com/erenbogazkesen/Turkish-Product-Review/blob/main/images/Örnek 1.cümle lime çıktısı.png?raw=true)

### 1. Cümle Entegre Gradyanlar Örneği

![Entegre Gradyanlar](https://github.com/erenbogazkesen/Turkish-Product-Review/blob/main/images/Örnek 1.cümle yöntem 2.png?raw=true)

### 1. Cümle Eli5 Çıktısı
![Eli5](https://github.com/erenbogazkesen/Turkish-Product-Review/blob/main/images/Örnek 1.cümle yöntem 1.png?raw=true)

### 2. Cümle LIME Örneği

![LIME Açıklaması](https://github.com/erenbogazkesen/Turkish-Product-Review/blob/main/images/Örnek 2.cümle lime çıktısı.png?raw=true)

### 2. Cümle Entegre Gradyanlar Örneği

![Entegre Gradyanlar](https://github.com/erenbogazkesen/Turkish-Product-Review/blob/main/images/Örnek 2.cümle yöntem 1.png?raw=true)

### 2. Cümle Eli5 Çıktısı
![Eli5](https://github.com/erenbogazkesen/Turkish-Product-Review/blob/main/images/Örnek 2.cümle yöntem 2.png?raw=true)

### 3. Cümle LIME Örneği

![LIME Açıklaması](https://github.com/erenbogazkesen/Turkish-Product-Review/blob/main/images/Örnek 3.cümle lime çıktısı.png?raw=true)

### 3. Cümle Entegre Gradyanlar Örneği

![Entegre Gradyanlar](https://github.com/erenbogazkesen/Turkish-Product-Review/blob/main/images/Örnek 3.cümle yöntem 1.png?raw=true)

### 3. Cümle Eli5 Çıktısı
![Eli5](https://github.com/erenbogazkesen/Turkish-Product-Review/blob/main/images/Örnek 3.cümle yöntem 2.png?raw=true)

 

