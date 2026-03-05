import os
import time
from datetime import datetime
from typing import Any

from ollama import Client


MODELS = [
    "gpt-oss:20b",
    "gpt-oss:120b",
    "qwen3-coder:480b",
]

TASKS = [
    {
        "id": "rag_explanation",
        "title": "RAG explanation for beginners",
        "prompt": (
            "Explain Retrieval-Augmented Generation (RAG) for beginners.\n"
            "Use this exact structure:\n"
            "1) Definition (1 sentence)\n"
            "2) Pipeline (3 bullet points)\n"
            "3) When to use (2 bullet points)\n"
            "4) One limitation (1 bullet point)\n"
            "Keep it practical and concise."
        ),
        "keywords": ["retrieval", "context", "knowledge", "hallucination"],
    },
    {
        "id": "rag_python",
        "title": "Mini Python RAG skeleton",
        "prompt": (
            "Write a minimal Python skeleton for a RAG pipeline using placeholders.\n"
            "Requirements:\n"
            "- Use function names: embed_query, retrieve_top_k, build_prompt, ask_model\n"
            "- Show data flow from query to final answer\n"
            "- Include comments for where vector DB and LLM calls happen\n"
            "- Keep it to 12-20 lines of code"
        ),
        "keywords": ["embed_query", "retrieve_top_k", "build_prompt", "ask_model"],
    },
]


def quality_score(text: str, required_keywords: list[str]) -> tuple[int, str]:
    lower = text.lower()
    keyword_hits = sum(1 for kw in required_keywords if kw.lower() in lower)
    keyword_score = int((keyword_hits / len(required_keywords)) * 60)

    # Format score rewards actionable structure and readability.
    bullets = text.count("- ")
    lines = max(1, len([line for line in text.splitlines() if line.strip()]))
    format_score = 0
    if bullets >= 2:
        format_score += 15
    if 4 <= lines <= 28:
        format_score += 15
    if len(text) >= 140:
        format_score += 10

    total = min(100, keyword_score + format_score)
    verdict = "Strong" if total >= 80 else "Usable" if total >= 60 else "Weak"
    return total, verdict


def run_task(client: Client, model: str, task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": task["prompt"]}],
            stream=False,
        )
        elapsed = time.perf_counter() - started
        text = response["message"]["content"].strip()
        score, verdict = quality_score(text, task["keywords"])
        return {
            "ok": True,
            "latency": elapsed,
            "score": score,
            "verdict": verdict,
            "answer": text,
            "error": "",
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "ok": False,
            "latency": elapsed,
            "score": 0,
            "verdict": "Failed",
            "answer": "",
            "error": str(exc),
        }


def print_summary(results: dict[str, dict[str, dict[str, Any]]]) -> None:
    print("\n=== Overall Summary ===")
    print("| Model | Success | Avg Latency (s) | Avg Score |")
    print("|---|---:|---:|---:|")
    for model, task_results in results.items():
        task_values = list(task_results.values())
        success_count = sum(1 for r in task_values if r["ok"])
        avg_latency = sum(r["latency"] for r in task_values) / len(task_values)
        avg_score = sum(r["score"] for r in task_values) / len(task_values)
        print(f"| {model} | {success_count}/{len(task_values)} | {avg_latency:.2f} | {avg_score:.1f} |")

    print("\n=== Per-Task Breakdown ===")
    print("| Task | Model | Latency (s) | Score | Verdict |")
    print("|---|---|---:|---:|---|")
    for task in TASKS:
        task_id = task["id"]
        for model in MODELS:
            item = results[model][task_id]
            print(
                f"| {task['title']} | {model} | {item['latency']:.2f} | "
                f"{item['score']} | {item['verdict']} |"
            )


def save_markdown_report(results: dict[str, dict[str, dict[str, Any]]], path: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# Ollama Cloud Model Comparison Report")
    lines.append("")
    lines.append(f"- Generated: {now}")
    lines.append(f"- Models: {', '.join(MODELS)}")
    lines.append(f"- Tasks: {len(TASKS)}")
    lines.append("")
    lines.append("## Overall Summary")
    lines.append("")
    lines.append("| Model | Success | Avg Latency (s) | Avg Score |")
    lines.append("|---|---:|---:|---:|")

    for model, task_results in results.items():
        task_values = list(task_results.values())
        success_count = sum(1 for r in task_values if r["ok"])
        avg_latency = sum(r["latency"] for r in task_values) / len(task_values)
        avg_score = sum(r["score"] for r in task_values) / len(task_values)
        lines.append(f"| {model} | {success_count}/{len(task_values)} | {avg_latency:.2f} | {avg_score:.1f} |")

    lines.append("")
    lines.append("## Detailed Outputs")
    lines.append("")

    for task in TASKS:
        task_id = task["id"]
        lines.append(f"### {task['title']}")
        lines.append("")
        lines.append("**Prompt**")
        lines.append("")
        lines.append("```text")
        lines.append(task["prompt"])
        lines.append("```")
        lines.append("")
        for model in MODELS:
            item = results[model][task_id]
            lines.append(f"#### {model}")
            lines.append("")
            lines.append(f"- Latency: {item['latency']:.2f}s")
            lines.append(f"- Score: {item['score']}")
            lines.append(f"- Verdict: {item['verdict']}")
            if item["error"]:
                lines.append(f"- Error: {item['error']}")
                lines.append("")
            else:
                lines.append("")
                lines.append("```text")
                lines.append(item["answer"])
                lines.append("```")
                lines.append("")

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main() -> None:
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("Set OLLAMA_API_KEY first.")

    client = Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    print("Running model comparison on structured tasks...")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Tasks: {', '.join(task['title'] for task in TASKS)}")

    results: dict[str, dict[str, dict[str, Any]]] = {model: {} for model in MODELS}

    for task in TASKS:
        print(f"\nTask: {task['title']}")
        for model in MODELS:
            item = run_task(client, model, task)
            results[model][task["id"]] = item
            print(
                f"  - {model}: {item['latency']:.2f}s | "
                f"score {item['score']} | {item['verdict']}"
            )
            if item["error"]:
                print(f"    error: {item['error']}")

    print_summary(results)

    report_path = "model_comparison_report.md"
    save_markdown_report(results, report_path)
    print(f"\nSaved full responses to: {report_path}")


if __name__ == "__main__":
    main()
