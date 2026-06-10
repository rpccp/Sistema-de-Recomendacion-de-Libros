# Sistema de Recomendacion de Libros

Sistema de recomendacion basado en embeddings, Ollama y FAISS. El proyecto toma un dataset de libros, transforma la informacion relevante de cada titulo en una representacion textual, genera embeddings y busca libros semanticamente similares mediante busqueda vectorial.

## Objetivo

El sistema permite ingresar un titulo o parte de un titulo y obtener recomendaciones de libros parecidos segun descripcion, autores, categorias, rating y cantidad de paginas. Esta pensado como una demostracion practica de recuperacion semantica de informacion aplicada a un catalogo de libros.

## Tecnologias

- Python
- Pandas y NumPy para procesamiento de datos
- Ollama para generar embeddings localmente
- FAISS para busqueda eficiente por similitud
- Jupyter Notebook para exploracion y demostracion

## Estructura

```text
.
|-- app.py                 # CLI para consultar, inspeccionar y regenerar el indice
|-- books.csv              # Dataset de libros
|-- indice                 # Indice FAISS ya generado
|-- index_metadata.json    # Metadatos del indice incluido
|-- requirements.txt       # Dependencias del proyecto
|-- sistema.ipynb          # Notebook de demostracion
`-- src/
    |-- __init__.py
    `-- recommender.py     # Logica principal del recomendador
```

## Instalacion

1. Crear y activar un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Instalar y ejecutar Ollama:

```bash
ollama serve
ollama pull deepseek-r1:7b
```

> El archivo `indice` incluido fue generado para embeddings de 3584 dimensiones. Si se usa otro modelo, hay que regenerar el indice con el mismo modelo que se vaya a usar para recomendar.

## Uso

Ver informacion del dataset y del indice:

```bash
python app.py info
```

Obtener recomendaciones:

```bash
python app.py recommend "Rich Dad Poor Dad" --k 5
```

Usar otro modelo o endpoint de Ollama:

```bash
python app.py --model deepseek-r1:7b --ollama-url http://localhost:11434 recommend "Gilead"
```

Regenerar el indice FAISS:

```bash
python app.py build-index
```

## Como funciona

1. Carga `books.csv` y valida que existan las columnas necesarias.
2. Limpia texto faltante y corrige errores comunes de encoding presentes en el dataset.
3. Construye una representacion textual por libro con titulo, autores, descripcion, categoria y metricas.
4. Genera embeddings con Ollama.
5. Usa FAISS para recuperar los vectores mas cercanos al libro elegido.
6. Devuelve los titulos mas similares junto con autor, categoria, rating y distancia vectorial.

## Estado del proyecto

- Incluye un indice FAISS precomputado para evitar regenerar todos los embeddings en cada ejecucion.
- La logica principal esta separada del notebook para que el proyecto sea mas facil de ejecutar, revisar y extender.
- La CLI agrega validaciones y mensajes de error para casos comunes: dataset faltante, indice incompatible, Ollama apagado o modelo no instalado.

## Posibles mejoras

- Agregar una interfaz web simple con Streamlit.
- Evaluar modelos especificos de embeddings y comparar calidad de recomendaciones.
- Incorporar filtros por categoria, autor, rating minimo o cantidad de paginas.
- Guardar metadatos del modelo junto al indice para detectar incompatibilidades automaticamente.

## Autor

Rocco Gasparini

Email: rgasparini75@gmail.com
