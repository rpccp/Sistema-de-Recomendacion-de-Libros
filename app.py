from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss

from src.recommender import BookRecommender, build_faiss_index, load_books


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sistema de recomendacion de libros con embeddings y FAISS."
    )
    parser.add_argument("--data", default="books.csv", help="Ruta al dataset CSV.")
    parser.add_argument("--index", default="indice", help="Ruta al indice FAISS.")
    parser.add_argument("--model", default="deepseek-r1:7b", help="Modelo de Ollama.")
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="URL base de Ollama.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    recommend = subparsers.add_parser("recommend", help="Recomendar libros por titulo.")
    recommend.add_argument("title", help="Titulo o fragmento del titulo favorito.")
    recommend.add_argument("--k", type=int, default=5, help="Cantidad de recomendaciones.")

    subparsers.add_parser("build-index", help="Regenerar el indice FAISS desde cero.")
    subparsers.add_parser("info", help="Mostrar informacion del dataset y el indice.")
    return parser


def print_recommendations(args: argparse.Namespace) -> None:
    recommender = BookRecommender(
        data_path=Path(args.data),
        index_path=Path(args.index),
        model=args.model,
        ollama_url=args.ollama_url,
    )
    recommendations = recommender.recommend(args.title, k=args.k)

    print(f"\nRecomendaciones para: {args.title}\n")
    for number, book in enumerate(recommendations, start=1):
        print(f"{number}. {book.title}")
        print(f"   Autor/es: {book.authors}")
        print(f"   Categoria: {book.categories}")
        print(f"   Rating: {book.average_rating}")
        print(f"   Distancia FAISS: {book.distance:.4f}")
        print()


def rebuild_index(args: argparse.Namespace) -> None:
    books = load_books(args.data)
    index = build_faiss_index(
        books["representacion_textual"],
        model=args.model,
        ollama_url=args.ollama_url,
    )
    faiss.write_index(index, args.index)
    print(f"Indice regenerado en {args.index} con {index.ntotal} libros.")


def print_info(args: argparse.Namespace) -> None:
    books = load_books(args.data)
    print(f"Dataset: {args.data}")
    print(f"Libros: {len(books)}")
    print(f"Columnas: {', '.join(books.columns)}")

    index_path = Path(args.index)
    if index_path.exists():
        index = faiss.read_index(str(index_path))
        print(f"Indice: {args.index}")
        print(f"Embeddings: {index.ntotal}")
        print(f"Dimensiones: {index.d}")

        metadata_path = Path("index_metadata.json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            print(f"Modelo registrado: {metadata.get('model', 'Sin datos')}")
    else:
        print(f"Indice: no existe en {args.index}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "recommend":
        print_recommendations(args)
    elif args.command == "build-index":
        rebuild_index(args)
    elif args.command == "info":
        print_info(args)


if __name__ == "__main__":
    main()
