import json
import sys
import argparse
from pathlib import Path
from tools.search_papers import search_papers


TOOLS = {
    "search_papers": search_papers,
}

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": "Search the machine learning literature for papers relevant to a research query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The literature search query."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}

TOOL_SCHEMAS = [TOOL_SCHEMA]


# Allow imports from the project root when running: python src/agent.py
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import MODEL_NAME, client


SYSTEM_PROMPT = """
You are a machine learning research assistant.
Your job is to investigate whether a proposed ML research idea may overlap with existing work.
For this first prototype, you have access to a paper-search tool.
Use the tool when it would help answer the user's research question.
Once you have enough information, provide a concise answer and mention what evidence you used.
Do not claim that an idea is definitely novel just because the search found nothing.
""".strip()


def execute_tool(tool_name: str, arguments: str) -> str:
    """Validate and execute a requested tool."""
    if tool_name not in TOOLS:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        args = json.loads(arguments)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid tool arguments: {exc}"})

    try:
        return TOOLS[tool_name](**args)
    except Exception as exc:
        return json.dumps({"error": f"Tool execution failed: {exc}"})


def run_agent(
    user_message: str,
    max_turns: int = 8,
    verbose: bool = False
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    if verbose:
        print("\n[User]")
        print(user_message)

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        assistant_message = {
            "role": "assistant",
            "content": message.content,
        }

        if message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]

        messages.append(assistant_message)

        if verbose:
            print("\n[Assistant]")
            print(assistant_message)

        if not message.tool_calls:
            return message.content or ""

        for call in message.tool_calls:
            result = execute_tool(
                call.function.name,
                call.function.arguments,
            )

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

            if verbose:
                print("\n[Tool]")
                print(f"tool_call_id: {call.id}")
                print(f"tool: {call.function.name}")
                print(f"result: {result}")

    return "Agent stopped because it reached the maximum number of turns."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show messages exchanged between the user, LLM, and tools."
    )
    args = parser.parse_args()

    idea = input("Research idea: ").strip()
    if not idea:
        raise SystemExit("Research idea cannot be empty.")

    print("\nAgent:\n")
    print(run_agent(idea, verbose=args.verbose))