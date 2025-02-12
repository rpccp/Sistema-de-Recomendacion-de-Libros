# Sistema de Recomendacion de Libros
## Acerca del proyecto

En este proyecto hice un sistema de recomendacion, en este caso, de libros, en Python con LLMs

### Que hace este programa?

Lo que hace este programa es tomar un data set de muchos libros distintos, en el que cada libro tiene distintos atributos (titulo, autor, rating, genero, etc...).

Toma esos atributos y construye una representacion textual, en este caso texto crudo.

Luego usa tecnologia LLM para convertir esa representacion textual en un vector (en este caso use DeepSeek R1 7b), el cual se va a guardar en una Vector Store.

Entonces, cuando elijo un libro y quiero recomendaciones, lo convierte en un vector y busca cuales libros son los mas parecidos preguntando cuales son los vectores mas cercanos (Similarity Search).

Luego devuelve los libros asociados a los vectores mas cercanos

### Construido con

* Python
* Ollama
* Pandas
* Faiss
* NumPy

### Que aprendi?

Construir este proyecto fue un desafio, en el cual tuve la oportunidad de aprender y mejorar mis habilidades en:

* Manipulacion de grandes cantidades de datos con herramientas de Ciencias de Datos
* Manipulacion de vectores grandes multidimencionales
* Uso de LLMs
* Uso de herramientas para busqueda eficiente por similitud

### Contacto

Rocco Gasparini - [@rtffrock](https://x.com/rtffrock) - rgasparini75@gmail.com