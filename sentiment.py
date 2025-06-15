!pip install -U transformers datasets torch numpy pandas matplotlib seaborn shap lime

# Orijinal veri setini al

dataset = load\_dataset("turkish\_product\_reviews")

# Eğitim verisi

train\_df = pd.DataFrame(dataset\["train"])

# Temizleme işleminden sonra metinleri yeni sütun olarak ekliyoruz

train\_df\["clean\_text"] = train\_df\["sentence"].apply(temizle\_metin)

# Olumlu ve olumsuzları ayırıyoruz

olumlu = train\_df\[train\_df\["sentiment"] == 1]
olumsuz = train\_df\[train\_df\["sentiment"] == 0]

# Rastgele 5000'er örnek seçimi

olumlu\_ornekler = olumlu.sample(5000, random\_state=42)
olumsuz\_ornekler = olumsuz.sample(5000, random\_state=42)

# Birleştirme ve karıştırma işlemi

dengeli\_dataset = pd.concat(\[olumlu\_ornekler, olumsuz\_ornekler]).sample(frac=1, random\_state=42).reset\_index(drop=True)

# Temiz metinleri ve etiketleri alma işlemi

texts = list(dengeli\_dataset\["clean\_text"])
labels = list(dengeli\_dataset\["sentiment"])

# Train-test ayırma işlemi

train\_texts, val\_texts, train\_labels, val\_labels = train\_test\_split(
texts,
labels,
test\_size=0.2,
random\_state=42
)

import re

def temizle\_metin(metin):
\# Küçük harfe çevir
metin = metin.lower()
\# Noktalama ve özel karakter temizleme işlemi
metin = re.sub(r"\[^a-zA-ZçğıöşüÇĞİÖŞÜ0-9\s]", "", metin)
\# Birden fazla boşlukları teke indir
metin = re.sub(r"\s+", " ", metin).strip()
return metin

import pandas as pd
import torch
from datasets import load\_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model\_selection import train\_test\_split

# Orijinal veri setini al

dataset = load\_dataset("turkish\_product\_reviews")

# Eğitim verisi

train\_df = pd.DataFrame(dataset\["train"])

# Temizlenmiş metinleri yeni sütun olarak ekle

train\_df\["clean\_text"] = train\_df\["sentence"].apply(temizle\_metin)

from sklearn.model\_selection import train\_test\_split
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch

# Tokenizer'ı yükle

tokenizer = AutoTokenizer.from\_pretrained("dbmdz/bert-base-turkish-uncased")

model\_path = "./turkish-sentiment-model"
trainer.save\_model(model\_path)
print(f"Model kaydedildi: {model\_path}")

# Metrikleri hesapla

from sklearn.metrics import accuracy\_score, precision\_score, recall\_score, f1\_score, confusion\_matrix

# predict sonrası örnek

preds = trainer.predict(val\_dataset)
y\_pred = preds.predictions.argmax(axis=1)
y\_true = val\_labels

# metrikler

accuracy = accuracy\_score(y\_true, y\_pred)
precision = precision\_score(y\_true, y\_pred)
recall = recall\_score(y\_true, y\_pred)
f1 = f1\_score(y\_true, y\_pred)

# Specificity hesaplama

tn, fp, fn, tp = confusion\_matrix(y\_true, y\_pred).ravel()
specificity = tn / (tn + fp)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score: {f1:.4f}")

import lime
from lime.lime\_text import LimeTextExplainer

# LIME için model tahmin fonksiyonu

def predict\_proba\_wrapper(texts):
\# Metinleri tokenize et
inputs = tokenizer(texts, padding=True, truncation=True, max\_length=128, return\_tensors="pt")

```
# Tahmin yap
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()

return probs
```

# Test metni

test\_text = wrong\_predictions.iloc\[25]\['Metin']  # Bir hatalı örnek üzerinde açıklama yap
print(f"Test metni: {test\_text}")

# LIME açıklayıcısını oluştur

explainer = LimeTextExplainer(class\_names=\["Olumsuz", "Olumlu"])

# Açıklama oluştur

explanation = explainer.explain\_instance(
test\_text,
predict\_proba\_wrapper,
num\_features=10,  # Kaç özelliğin gösterileceği
num\_samples=1000  # Perturbasyon sayısı
)

# Sonuçları görselleştir

explanation.show\_in\_notebook(text=test\_text)



explanation.save\_to\_file('lime\_aciklama.html')

# Önemli kelimeleri çıkar

lime\_weights = dict(explanation.as\_list())
print("LIME'a göre önemli kelimeler ve ağırlıkları:")
print(lime\_weights)

from captum.attr import LayerIntegratedGradients
import torch

# Modeli değerlendirme moduna getir

model.eval()

# Test metni

test\_text = wrong\_predictions.iloc\[0]\['Metin']
encoded\_input = tokenizer(test\_text, padding=True, truncation=True, max\_length=128, return\_tensors="pt")

# Token ID'leri ve Attention Mask'in tensor olduğundan emin ol

input\_ids = encoded\_input\['input\_ids']
attention\_mask = encoded\_input\['attention\_mask']

# Sınıf için tahmin fonksiyonu (Olumlu sınıf için)

def forward\_func(input\_ids, attention\_mask):
outputs = model(input\_ids=input\_ids, attention\_mask=attention\_mask)
return outputs.logits\[:, 1]  # Pozitif sınıf (Olumlu)

# Modelin embedding katmanı

ref\_token\_id = tokenizer.pad\_token\_id  # PAD token as reference
sep\_token\_id = tokenizer.sep\_token\_id
cls\_token\_id = tokenizer.cls\_token\_id

# Referans oluştur

ref\_input\_ids = torch.tensor(\[\[cls\_token\_id] + \[ref\_token\_id] \* (input\_ids.shape\[1] - 2) + \[sep\_token\_id]],
dtype=torch.long)
ref\_attention\_mask = torch.zeros\_like(attention\_mask)

# LayerIntegratedGradients ile

lig = LayerIntegratedGradients(forward\_func, model.bert.embeddings)

# Attribution hesapla

attributions, delta = lig.attribute(
inputs=(input\_ids, attention\_mask),
baselines=(ref\_input\_ids, ref\_attention\_mask),
return\_convergence\_delta=True,
internal\_batch\_size=4,
n\_steps=50
)

# Token'ları çıkar

tokens = tokenizer.convert\_ids\_to\_tokens(input\_ids\[0])

# Önem değerlerini topla

attributions\_sum = attributions.sum(dim=-1).detach().numpy()

# Görselleştirme

print(f"Test metni: {test\_text}")
print("\nToken'lar ve önem değerleri:")
for token, importance in zip(tokens, attributions\_sum\[0]):
print(f"{token}: {importance:.4f}")

# En önemli token'ları göster

token\_importance = list(zip(tokens, attributions\_sum\[0]))
sorted\_importance = sorted(token\_importance, key=lambda x: abs(x\[1]), reverse=True)
print("\nEn önemli 10 token:")
for token, importance in sorted\_importance\[:10]:
print(f"{token}: {importance:.4f}")

import eli5
from eli5.lime import TextExplainer
import numpy as np

te = TextExplainer(random\_state=42)

def predict\_proba\_eli5(text):
inputs = tokenizer(\[text], padding=True, truncation=True, max\_length=128, return\_tensors="pt")
with torch.no\_grad():
outputs = model(\*\*inputs)
probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()\[0]
return probs

# Test metni

test\_text = wrong\_predictions\['Metin'].iloc\[0]
print(f"Test metni: {test\_text}")

# fit fonksiyonu

te.fit(test\_text, lambda x: np.array(\[predict\_proba\_eli5(t) for t in x]))
explanation = te.explain\_prediction(target\_names=\["Olumsuz", "Olumlu"])

# Göster

print(eli5.format\_as\_text(explanation))