# Resume Artículos Grandes

Este script de Python, diseñado para ejecutarse en Google Colab, utiliza un modelo de lenguaje grande (LLM) y la biblioteca `langchain` para resumir artículos extensos sobre tecnologías e IA. El script carga un artículo desde una URL, lo divide en fragmentos manejables, genera resúmenes parciales de cada fragmento y finalmente combina estos resúmenes en un resumen coherente y completo.

## Requisitos

- **Google Colab**: Este script está diseñado para ejecutarse en Google Colab.
- **API Key de OpenAI**: Necesitarás una clave de API de OpenAI para utilizar el modelo de lenguaje.

## Configuración

1. **Instalar Dependencias**:
   - Ejecuta las siguientes celdas en Google Colab para instalar las dependencias necesarias:
     ```python
     !pip install langchain
     !pip install langchain_community
     !pip install openai
     !pip install tiktoken
     ```

2. **Configurar la Clave de API de OpenAI**:
   - Crea un archivo `.env` en tu entorno de Google Colab y agrega tu clave de API de OpenAI:
     ```
     OPENAI_API_KEY=tu_clave_de_api_de_openai
     ```
   - Alternativamente, puedes configurar la clave de API directamente en el script.

## Uso

1. **Cargar el Script**:
   - Copia y pega el contenido del script en una celda de Google Colab.

2. **Ejecutar el Script**:
   - Ejecuta la celda para cargar el artículo, dividirlo en fragmentos, generar resúmenes parciales y combinar estos resúmenes en un resumen final.

3. **Ingresar la URL del Artículo**:
   - El script te pedirá que ingreses la URL del artículo que deseas resumir. Puedes usar una de las URLs de ejemplo proporcionadas o ingresar una URL diferente.

## Ejemplo de Uso

1. **Ejecutar el Script**:
   - Ejecuta la celda que contiene el script.

2. **Ingresar la URL**:
   - El script te pedirá que ingreses la URL del artículo. Puedes copiar y pegar una de las URLs de ejemplo o ingresar una URL diferente.

3. **Ver el Resumen**:
   - El script cargará el artículo, lo dividirá en fragmentos, generará resúmenes parciales y combinará estos resúmenes en un resumen final, que se mostrará en la consola.

## Ejemplo de URLs

Aquí tienes algunas URLs de ejemplo que puedes usar para probar el script:

1. [Análisis de las expectativas en IA para 2025 según BBC Mundo](https://www.bbc.com/mundo/articles/c4gxzx0kpp6o)
2. [Crecimiento y aplicaciones de la IA generativa en 2025](https://www.computerworld.es/article/3631423/como-sera-el-avance-de-la-inteligencia-artificial-generativa-en-2025.html)
3. [Tendencias emergentes en IA y su impacto en la investigación científica](https://codelabsacademy.com/es/blog/artificial-intelligence-trends-in-2025-whats-next-in-ai)
4. [Integración de la IA en la computación en la nube](https://baufest.com/el-futuro-de-la-ia-y-la-computacion-en-la-nube-tendencias-para-2025-y-mas-alla/)
5. [Discusiones sobre IA en el Foro Económico Mundial de Davos 2025](https://es.weforum.org/stories/2025/01/ia-tecnologia-y-la-era-inteligente-en-davos-2025-lo-que-hay-que-saber/)
6. [Avances esperados en IA y su impacto económico](https://www.ironhack.com/es/blog/avances-en-inteligencia-artificial-una-mirada-2024)
7. [Cinco tendencias clave en IA para 2025](https://bimsoluciones.com/5-tendencias-ia-para-2025-que-no-puedes-perderte/)
8. [Desafíos y oportunidades en el desarrollo de la IA en 2025](https://www.next-step.es/tendencias-y-retos-de-la-ia-en-2025/)
9. [Transformaciones de la IA en educación y aprendizaje](https://sergio.ec/ia-2025-el-futuro-que-ya-esta-aqui/)
10. [Proyecto europeo de modelos de lenguaje abiertos y transparentes](https://cadenaser.com/comunitat-valenciana/2025/02/05/la-empresa-prompsit-del-pcumh-participa-en-un-proyecto-sobre-modelos-de-lenguaje-masivos-de-codigo-abierto-para-una-ia-transparente-en-europa-radio-elche/)

## Notas

- **Tamaño del Fragmento**: Puedes ajustar el tamaño del fragmento (`chunk_size`) y el solapamiento (`chunk_overlap`) según tus necesidades.
- **Modelo de LLM**: Puedes cambiar el modelo de LLM (`gpt-4-turbo`, `gpt-3.5-turbo`, etc.) según tus preferencias y disponibilidad.

