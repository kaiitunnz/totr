import argparse
import re


def convert_file(input_file, output_file, system_prompt=None, direct=False):
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    parts = re.split(r"(# METADATA: \{.*?\})", content)

    examples = []
    current_example = None

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("# METADATA:"):
            if current_example is not None:
                examples.append(current_example)
            current_example = {"metadata": part, "content": ""}
        elif current_example is not None:
            current_example["content"] += part + "\n"

    # Add the last example if there is one
    if current_example is not None:
        examples.append(current_example)

    # Convert each example to the llama-instruct format
    converted_examples = []

    for example in examples:
        metadata = example["metadata"]
        content_text = example["content"]

        q_match = re.search(r"Q:(.*?)(?=A:|$)", content_text, re.DOTALL)
        a_match = re.search(r"A:(.*?)$", content_text, re.DOTALL)

        assert q_match is not None and a_match is not None, f"Content: {content_text}"

        question = q_match.group(1).strip()
        full_answer = a_match.group(1).strip()

        context_end = content_text.find("Q:")
        context = content_text[:context_end].strip() if context_end > 0 else ""

        # Format the converted example
        converted_example = [
            metadata,
            "",
            "# Context:",
            context,
            "",
            "#Question:",
            question,
            "",
            "#Answer:",
            "The answer is: " + full_answer,
        ]

        converted_examples.append("\n".join(converted_example))

    output_content = "\n\n".join(converted_examples)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_content)

    print(f"Converted {len(examples)} examples to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert prompts to llama-instruct format with no reasoning"
    )
    parser.add_argument("--input", "-i", required=True, help="Input file")
    parser.add_argument("--output", "-o", required=True, help="Output file")
    parser.add_argument("--system-prompt", "-s", help="System prompt to add")
    parser.add_argument(
        "--direct", action="store_true", help="Direct mode (no modifications)"
    )

    args = parser.parse_args()

    convert_file(args.input, args.output, args.system_prompt, args.direct)


if __name__ == "__main__":
    main()
