# Procesamiento de Lenguaje Natural Version 2

**Ejercicio: Procesamiento de Lenguaje Natural (NLP)**

Utiliza Python y una librería como Hugging Face Transformers para entrenar un modelo que realice una tarea de clasificación de texto (por ejemplo, clasificación de opiniones: positiva o negativa). Proporciona un conjunto de datos de ejemplo (puede ser un pequeño conjunto de datos de reseñas) y demuestra cómo entrenar el modelo, evaluarlo y ajustarlo.

**Respuesta:**

Este script utiliza Python y la librería Hugging Face Transformers para entrenar un modelo de clasificación de texto. En este caso, el modelo se entrena para clasificar reseñas de películas como positivas o negativas utilizando el conjunto de datos IMDB.

Además, se integra Weights & Biases (W&B) para registrar métricas, hiperparámetros y artefactos durante el entrenamiento y la evaluación del modelo.

**Breve comentario sobre Weights & Biases (W&B)**

Weights & Biases (W&B) es una plataforma de experimentación para machine learning que permite registrar, visualizar y comparar experimentos. Al integrar W&B en este script, podemos aprovechar varias funciones y ventajas:

1. **Registro de Métricas**: W&B registra automáticamente métricas como la precisión y la pérdida durante el entrenamiento y la evaluación del modelo. Esto permite un seguimiento detallado del rendimiento del modelo a lo largo del tiempo.

2. **Registro de Hiperparámetros**: Los hiperparámetros configurados al iniciar el experimento (como la tasa de aprendizaje, el tamaño del lote y el número de épocas) se registran automáticamente. Esto facilita la reproducción de experimentos y la comparación de diferentes configuraciones.

3. **Visualización de Resultados**: W&B proporciona gráficos y paneles interactivos para visualizar el rendimiento del modelo. Esto incluye gráficos de pérdida y precisión, así como visualizaciones de hiperparámetros.

4. **Comparación de Experimentos**: W&B permite comparar fácilmente diferentes experimentos y configuraciones de modelos. Esto es útil para identificar qué configuraciones de hiperparámetros o qué modelos preentrenados funcionan mejor para la tarea específica.

5. **Colaboración**: W&B facilita el trabajo en equipo al permitir compartir experimentos y resultados con otros colaboradores. Esto es especialmente útil en proyectos de investigación o desarrollo colaborativo.

6. **Registro de Artefactos**: Los modelos guardados durante el entrenamiento se registran como artefactos en W&B. Esto permite un seguimiento claro de las versiones del modelo y facilita la gestión de modelos.

## Paso 1: Instalación de Dependencias

Primero, instala las librerías necesarias:

```python
!pip install python-dotenv
!pip install transformers datasets torch
!pip install wandb  # Instalar Weights & Biases
```

## Paso 2: Cargar la API Token de Hugging Face

```python
from dotenv import load_dotenv, find_dotenv
import os

# Verifica y carga el archivo con las variables de entorno si existe
if load_dotenv(find_dotenv()):
    if "HF_TOKEN" in os.environ:  # Verifica que existe la variable HF_KEY
        print("HUGGING FACE API TOKEN cargada")  # Imprime el mensaje si existe la variable de OpenAI
    else:
        print("No se cargó la api_key de HUGGING FACE, deberá cargarla manualmente")
else:
    print("No se cargó la api_key, deberá cargarla manualmente")
```

## Paso 3: Inicializar Weights & Biases (W&B)

```python
import wandb

# Iniciar un nuevo experimento en W&B
wandb.init(project="text-classification-imdb", config={
    "learning_rate": 2e-5,
    "batch_size": 16,
    "epochs": 3,
    "weight_decay": 0.01,
})
```

## Paso 4: Cargar un Conjunto de Datos

Utilizaremos el conjunto de datos IMDB, que contiene reseñas de películas etiquetadas como positivas (1) o negativas (0). Este conjunto de datos está disponible en la librería datasets de Hugging Face.

```python
from datasets import load_dataset

# Cargar el conjunto de datos IMDB
dataset = load_dataset("imdb")

# Ver la estructura del dataset
print(dataset)
```

## Paso 5: Preprocesamiento de los Datos

Tokenizamos el texto utilizando un tokenizador preentrenado de Hugging Face. En este caso, usaremos el modelo "distilbert-base-uncased".

**Lista de modelos para usar:**

1. **BERT (Bidirectional Encoder Representations from Transformers)**:
   - `bert-base-uncased`: La versión base de BERT sin distinción de mayúsculas y minúsculas.
   - `bert-large-uncased`: Una versión más grande de BERT sin distinción de mayúsculas y minúsculas.
   - `bert-base-cased`: La versión base de BERT con distinción de mayúsculas y minúsculas.
   - `bert-large-cased`: Una versión más grande de BERT con distinción de mayúsculas y minúsculas.

2. **RoBERTa (Robustly Optimized BERT approach)**:
   - `roberta-base`: La versión base de RoBERTa.
   - `roberta-large`: Una versión más grande de RoBERTa.

3. **AlBERT (A Lite BERT)**:
   - `albert-base-v2`: La versión base de AlBERT.
   - `albert-large-v2`: Una versión más grande de AlBERT.

4. **XLNet**:
   - `xlnet-base-cased`: La versión base de XLNet con distinción de mayúsculas y minúsculas.
   - `xlnet-large-cased`: Una versión más grande de XLNet con distinción de mayúsculas y minúsculas.

5. **ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)**:
   - `electra-base-uncased`: La versión base de ELECTRA sin distinción de mayúsculas y minúsculas.
   - `electra-large-uncased`: Una versión más grande de ELECTRA sin distinción de mayúsculas y minúsculas.

6. **DeBERTa (Decoding-enhanced BERT with disentangled attention)**:
   - `deberta-base`: La versión base de DeBERTa.
   - `deberta-large`: Una versión más grande de DeBERTa.

7. **Longformer**:
   - `longformer-base-4096`: La versión base de Longformer, diseñada para manejar secuencias largas.
   - `longformer-large-4096`: Una versión más grande de Longformer.

8. **BigBird**:
   - `bigbird-roberta-base`: La versión base de BigBird, diseñada para manejar secuencias largas.
   - `bigbird-roberta-large`: Una versión más grande de BigBird.

9. **TinyBERT**:
   - `tiny-bert-base-uncased`: Una versión más pequeña y rápida de BERT sin distinción de mayúsculas y minúsculas.

10. **MobileBERT**:
    - `mobilebert-uncased`: Una versión más pequeña y rápida de BERT sin distinción de mayúsculas y minúsculas.

```python
from transformers import AutoTokenizer

# Modelos posibles a usar listados arriba
modelo = "distilbert-base-uncased"

# Cargar el tokenizador
tokenizer = AutoTokenizer.from_pretrained(modelo)

# Función para tokenizar el texto
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

# Aplicar tokenización al dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)
```

## Paso 6: Dividir el Conjunto de Datos

Dividimos el conjunto de datos en entrenamiento y evaluación.

```python
# Dividir el dataset en entrenamiento y evaluación
train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(100))  # Usamos solo 100 muestras para entrenamiento rápido
eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(10))    # Usamos 10 muestras para evaluación
```

## Paso 7: Entrenar el Modelo

Cargamos un modelo preentrenado y lo ajustamos para la tarea de clasificación.

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Cargar el modelo preentrenado
model = AutoModelForSequenceClassification.from_pretrained(modelo, num_labels=2)

# Configurar los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=wandb.config.learning_rate,  # Usar el learning_rate configurado en W&B
    per_device_train_batch_size=wandb.config.batch_size,  # Usar el batch_size configurado en W&B
    per_device_eval_batch_size=wandb.config.batch_size,
    num_train_epochs=wandb.config.epochs,  # Usar el número de épocas configurado en W&B
    weight_decay=wandb.config.weight_decay,  # Usar el weight_decay configurado en W&B
    logging_dir="./logs",  # Directorio para guardar logs
    logging_steps=10,  # Registrar métricas cada 10 pasos
    report_to="wandb",  # Enviar métricas a W&B
)

# Crear el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Entrenar el modelo
trainer.train()
```

## Paso 8: Evaluar el Modelo

Evaluamos el modelo en el conjunto de evaluación.

```python
# Evaluar el modelo
eval_results = trainer.evaluate()
print(f"Loss: {eval_results['eval_loss']}")

# Registrar la pérdida en W&B
wandb.log({"Loss": eval_results["eval_loss"]})
```

## Paso 9: Realizar Predicciones

Podemos usar el modelo entrenado para clasificar nuevas reseñas.

```python
import torch

# Verificar si hay una GPU disponible y mover el modelo al dispositivo adecuado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Función para predecir la clase de un texto
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Mover los tensores al mismo dispositivo
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)
    return "Positiva" if predictions.item() == 1 else "Negativa"

# Probamos el modelo entrenado con una lista de comentarios
comentarios = [
    "I loved this movie, it was amazing!",
    "Terrible film, I hated it.",
    "The acting was great, but the plot was boring.",
    "Absolutely fantastic!",
    "Worst movie ever.",
    "The movie was horrible, terribly bad.",
    "This movie was fantastic! I loved every minute of it."
]

for i, comentario in enumerate(comentarios):
    predict = predict_sentiment(comentario)
    print(f"Comentario {i + 1}:", comentario)
    print(f"Predicción: {predict}", "\n")
    wandb.log({f"Comentario {i + 1}": comentario})
    wandb.log({f"Predicción {i + 1}": predict})

# Finalizar el experimento en W&B e imprimir los parámetros
wandb.finish()
```

## Paso 10: Ajustar el Modelo (Opcional)

Si el rendimiento no es satisfactorio, es posible ajustar los hiperparámetros (por ejemplo, `learning_rate`, `num_train_epochs`, `batch_size`) o probar con otro modelo preentrenado.



## Gráficos de Evolución del Entrenamiento

Puedes ver los gráficos de evolución del entrenamiento en el siguiente enlace:

[Gráficos de Evolución del Entrenamiento](https://wandb.ai/alfredolawler-lawer-technology/huggingface/runs/pzhtx365?nw=nwuseralfredolawler)

```python
# Reiniciar el registro de parámetros en W&B
wandb.init()
