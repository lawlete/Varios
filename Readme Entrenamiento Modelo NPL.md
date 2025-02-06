Aquí tienes un archivo `README.md` basado en el código proporcionado:

---

# Procesamiento de Lenguaje Natural (NLP) con Hugging Face Transformers

Este proyecto demuestra cómo utilizar Python y la librería **Hugging Face Transformers** para entrenar un modelo de clasificación de texto. El objetivo es clasificar reseñas de películas como positivas o negativas utilizando el conjunto de datos **IMDB**.

## Requisitos

Antes de comenzar, asegúrate de tener instaladas las siguientes dependencias:

```bash
pip install python-dotenv transformers datasets torch
```

## Pasos del Proyecto

### Paso 1: Instalación de dependencias
Instala las librerías necesarias para el proyecto:

```bash
pip install python-dotenv transformers datasets torch
```

### Paso 2: Cargar la API Token de Hugging Face
Carga la API Token de Hugging Face desde un archivo `.env`:

```python
from dotenv import load_dotenv, find_dotenv
import os

if load_dotenv(find_dotenv()):
    if "HF_TOKEN" in os.environ:
        print("HUGGING FACE API TOKEN cargada")
    else:
        print("No se cargó la api_key de HUGGING FACE, deberá cargarla manualmente")
else:
    print("No se cargó la api_key, deberá cargarla manualmente")
```

### Paso 3: Cargar el conjunto de datos
Utilizamos el conjunto de datos **IMDB** para entrenar el modelo:

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
print(dataset)
```

### Paso 4: Preprocesamiento de los datos
Tokenizamos el texto utilizando el tokenizador de **DistilBERT**:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
```

### Paso 5: Dividir el conjunto de datos
Dividimos el conjunto de datos en entrenamiento y evaluación:

```python
train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(100))
```

### Paso 6: Entrenar el modelo
Cargamos un modelo preentrenado y lo ajustamos para la tarea de clasificación:

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

### Paso 7: Evaluar el modelo
Evaluamos el modelo en el conjunto de evaluación:

```python
eval_results = trainer.evaluate()
print(f"Precisión: {eval_results['eval_accuracy']}")
```

### Paso 8: Realizar predicciones
Usamos el modelo entrenado para clasificar nuevas reseñas:

```python
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)
    return "Positiva" if predictions == 1 else "Negativa"

review = "This movie was fantastic! I loved every minute of it."
print(f"Predicción: {predict_sentiment(review)}")
```

### Paso 9: Ajustar el modelo (opcional)
Si el rendimiento no es satisfactorio, puedes ajustar los hiperparámetros o probar con otro modelo preentrenado.

### Conjunto de datos de ejemplo reducido
Si prefieres usar un conjunto de datos más pequeño, puedes crear uno manualmente:

```python
import pandas as pd

data = {
    "text": [
        "I loved this movie, it was amazing!",
        "Terrible film, I hated it.",
        "The acting was great, but the plot was boring.",
        "Absolutely fantastic!",
        "Worst movie ever."
    ],
    "label": [1, 0, 0, 1, 0]  # 1 = Positiva, 0 = Negativa
}

df = pd.DataFrame(data)
print(df)
```

## Resultados
El modelo entrenado debería ser capaz de clasificar reseñas como positivas o negativas con una precisión razonable. Puedes mejorar el rendimiento utilizando un conjunto de datos más grande o ajustando los hiperparámetros.

---

Este `README.md` proporciona una guía clara y estructurada para reproducir el proyecto. ¡Espero que te sea útil!
