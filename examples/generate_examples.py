import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chat import build_session

SECRET_DETAIL = "My dog's name is Nimbus and I live in Austin."
RECALL_QUESTION = "What is my dog's name?"

FILLER_TOPICS = [
    "Tell me a fact about the ocean.",
    "What's a good recipe for pancakes?",
    "Explain how rainbows form.",
    "What's the tallest mountain in Africa?",
    "Give me a tip for better sleep.",
    "What's an interesting fact about octopuses?",
    "How does a refrigerator work?",
    "What's the history of the printing press?",
    "Suggest a beginner workout routine.",
    "What causes seasons to change?",
]


def run_conversation(memory_kind: str, turns: int = 34) -> str:
    if memory_kind == "hybrid":
        os.environ["MEMORY_MAX_TOKEN_LIMIT"] = "300"  # force summarization within this demo

    session = build_session(memory_kind)
    lines = [f"# {memory_kind} memory -- sample conversation\n"]

    def log(role: str, text: str) -> None:
        lines.append(f"**{role}:** {text}\n")

    log("You", SECRET_DETAIL)
    reply = session.send(SECRET_DETAIL)
    log("AI", reply)
    time.sleep(1)  # stay under Groq free-tier rate limit

    for i in range(turns):
        topic = FILLER_TOPICS[i % len(FILLER_TOPICS)]
        log("You", topic)
        reply = session.send(topic)
        log("AI", reply)
        time.sleep(1)

    log("You", RECALL_QUESTION)
    answer = session.send(RECALL_QUESTION)
    log("AI", answer)

    stats = session.stats()
    lines.append("\n## Final memory stats\n")
    lines.append(f"```\n{json.dumps(stats, indent=2)}\n```\n")
    passed = "PASSED" if "nimbus" in answer.lower() else "FAILED"
    lines.append(f"\n**Recall check:** {passed} -- expected 'Nimbus' in the final answer.\n")

    return "\n".join(lines)


def main() -> None:
    out_dir = os.path.dirname(__file__)

    try:
        print("Running buffer memory conversation (36 turns)...")
        buffer_md = run_conversation("buffer")
        with open(os.path.join(out_dir, "buffer_conversation.md"), "w", encoding="utf-8") as f:
            f.write(buffer_md)

        print("Running hybrid memory conversation (36 turns)...")
        hybrid_md = run_conversation("hybrid")
        with open(os.path.join(out_dir, "hybrid_conversation.md"), "w", encoding="utf-8") as f:
            f.write(hybrid_md)

        print("Done. See examples/buffer_conversation.md and examples/hybrid_conversation.md")
    except Exception as e:
        print(f"Failed to generate examples: {e}")
        print("Check that GROQ_API_KEY is set in .env and GROQ_MODEL is a currently valid model.")
        sys.exit(1)


if __name__ == "__main__":
    main()
