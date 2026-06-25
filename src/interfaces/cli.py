import typer
import requests
import json

from src.config import settings

app = typer.Typer()

API_URL = settings.api_url


def pretty(data):
    typer.echo(json.dumps(data, indent=2, ensure_ascii=False))


@app.command()
def upload(pdf: str):
    with open(pdf, "rb") as f:
        r = requests.post(
            f"{API_URL}/upload",
            files={"file": f},
        )

    pretty(r.json())


@app.command()
def ask(
    question: str,
    k: int = 5,
):
    r = requests.post(
        f"{API_URL}/ask",
        json={
            "question": question,
            "k": k,
        },
    )

    data = r.json()

    typer.echo("\n=== ANSWER ===\n")
    typer.echo(data["answer"])

    typer.echo("\n=== SOURCES ===\n")

    for c in data["citations"]:
        typer.echo(
            f"{c['source_marker']} "
            f"{c['filename']} "
            f"(page {c['page']})"
        )


@app.command()
def summarize(
    document: str | None = None,
    query: str | None = None,
):
    r = requests.post(
        f"{API_URL}/summarize",
        json={
            "document": document,
            "query": query,
        },
    )

    data = r.json()

    typer.echo("\n=== SUMMARY ===\n")
    typer.echo(data["summary"])

    typer.echo("\n=== KEY POINTS ===\n")

    for item in data["key_points"]:
        typer.echo(f"• {item}")


@app.command()
def quiz(
    document: str | None = None,
    query: str | None = None,
):
    r = requests.post(
        f"{API_URL}/quiz",
        json={
            "document": document,
            "query": query,
        },
    )

    data = r.json()

    for idx, q in enumerate(data["items"], 1):
        typer.echo(f"Question {idx}: {q['question']}")
        for i, opt in enumerate(q['options']):
            typer.echo(f"  {chr(65+i)}: {opt}")
        typer.echo(f"=> Correct Answer: {chr(65+q['correct_index'])}")
        typer.echo(f"=> Explanation: {q['explanation']}")
        typer.echo("-" * 30)


@app.command()
def flashcards(
    document: str | None = None,
    query: str | None = None,
):
    r = requests.post(
        f"{API_URL}/flashcard",
        json={
            "document": document,
            "query": query,
        },
    )

    data = r.json()
    for idx, card in enumerate(data["cards"], 1):
        typer.echo(f"Card {idx}:")
        typer.echo(f"  Front (Question): {card['front']}")
        typer.echo(f"  Back (Answer)   : {card['back']}")

        if card.get("hint"):
            typer.echo(f"  Hint            : {card['hint']}")

        typer.echo("-" * 30)

if __name__ == "__main__":
    app()