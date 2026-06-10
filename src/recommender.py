from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import pandas as pd
import requests


REQUIRED_COLUMNS = {
    "title",
    "authors",
    "categories",
    "description",
    "published_year",
    "average_rating",
    "num_pages",
}


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return "Sin datos"

    text = str(value).strip()
    if not text:
        return "Sin datos"

    # Fix common UTF-8-as-Latin-1 mojibake without touching already clean text.
    if "\u00c3" in text or "\u00e2" in text:
        try:
            return text.encode("latin1").decode("utf-8")
        except UnicodeError:
            return text

    return text


def load_books(path: str | Path) -> pd.DataFrame:
    books_path = Path(path)
    if not books_path.exists():
        raise FileNotFoundError(f"No se encontro el dataset: {books_path}")

    books = pd.read_csv(books_path)
    missing = REQUIRED_COLUMNS.difference(books.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Faltan columnas obligatorias en el CSV: {missing_list}")

    for column in books.select_dtypes(include=["object"]).columns:
        books[column] = books[column].map(_clean_text)

    books["representacion_textual"] = books.apply(build_text_representation, axis=1)
    return books


def build_text_representation(row: pd.Series) -> str:
    return "\n".join(
        [
            f"Titulo: {_clean_text(row['title'])}",
            f"Autores: {_clean_text(row['authors'])}",
            f"Descripcion: {_clean_text(row['description'])}",
            f"Categorias: {_clean_text(row['categories'])}",
            f"Anio de publicacion: {_clean_text(row['published_year'])}",
            f"Rating promedio: {_clean_text(row['average_rating'])}",
            f"Numero de paginas: {_clean_text(row['num_pages'])}",
        ]
    )


def embed_text(
    text: str,
    *,
    model: str,
    ollama_url: str = "http://localhost:11434",
    timeout: int = 120,
) -> np.ndarray:
    endpoint = f"{ollama_url.rstrip('/')}/api/embeddings"
    try:
        response = requests.post(
            endpoint,
            json={"model": model, "prompt": text},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "No se pudo obtener el embedding desde Ollama. "
            "Verifica que Ollama este corriendo y que el modelo este instalado."
        ) from exc

    payload = response.json()
    embedding = payload.get("embedding")
    if not embedding:
        raise RuntimeError(f"Ollama no devolvio un embedding valido para el modelo {model!r}.")

    return np.array(embedding, dtype="float32")


def build_faiss_index(
    texts: Iterable[str],
    *,
    model: str,
    ollama_url: str = "http://localhost:11434",
) -> faiss.IndexFlatL2:
    embeddings = [
        embed_text(text, model=model, ollama_url=ollama_url)
        for text in texts
    ]
    if not embeddings:
        raise ValueError("No hay textos para indexar.")

    matrix = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return index


@dataclass
class Recommendation:
    title: str
    authors: str
    categories: str
    average_rating: str
    distance: float
    description: str


@dataclass
class BookRecommender:
    data_path: Path = Path("books.csv")
    index_path: Path = Path("indice")
    model: str = "deepseek-r1:7b"
    ollama_url: str = "http://localhost:11434"

    def __post_init__(self) -> None:
        self.books = load_books(self.data_path)
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"No se encontro el indice FAISS en {self.index_path}. "
                "Ejecuta `python app.py build-index` para generarlo."
            )
        self.index = faiss.read_index(str(self.index_path))
        if self.index.ntotal != len(self.books):
            raise ValueError(
                "El indice FAISS y el dataset no tienen la misma cantidad de filas: "
                f"{self.index.ntotal} embeddings vs {len(self.books)} libros."
            )

    def find_book(self, query: str) -> pd.Series:
        matches = self.books[
            self.books["title"].str.contains(query, case=False, na=False, regex=False)
        ]
        if matches.empty:
            raise ValueError(f"No se encontraron libros con el titulo: {query}")
        return matches.iloc[0]

    def recommend(self, title: str, k: int = 5) -> list[Recommendation]:
        selected = self.find_book(title)
        query_embedding = embed_text(
            selected["representacion_textual"],
            model=self.model,
            ollama_url=self.ollama_url,
        ).reshape(1, -1)

        if query_embedding.shape[1] != self.index.d:
            raise ValueError(
                "La dimension del embedding no coincide con el indice FAISS: "
                f"{query_embedding.shape[1]} vs {self.index.d}. "
                "Usa el mismo modelo con el que se genero el indice o regeneralo."
            )

        distances, indices = self.index.search(query_embedding, k + 1)
        recommendations: list[Recommendation] = []
        for distance, idx in zip(distances.flatten(), indices.flatten()):
            if idx < 0 or idx == selected.name:
                continue
            row = self.books.iloc[int(idx)]
            recommendations.append(
                Recommendation(
                    title=_clean_text(row["title"]),
                    authors=_clean_text(row["authors"]),
                    categories=_clean_text(row["categories"]),
                    average_rating=_clean_text(row["average_rating"]),
                    distance=float(distance),
                    description=_clean_text(row["description"]),
                )
            )
            if len(recommendations) == k:
                break

        return recommendations

    def rebuild_index(self) -> None:
        index = build_faiss_index(
            self.books["representacion_textual"],
            model=self.model,
            ollama_url=self.ollama_url,
        )
        faiss.write_index(index, str(self.index_path))
        self.index = index
