import json


def transform_entry(entry):
    transformed = "# Context\n"

    # Add each piece of evidence
    for evidence in entry.get("evidence_list", []):
        transformed += f"# Title: {evidence.get('title', 'No title')}\n"
        transformed += f"# Fact: {evidence.get('fact', 'No fact')}\n\n"

    # Add the question and answer
    transformed += (
        f"Q: Answer the following question.\n{entry.get('query', 'No query')}\n"
    )
    transformed += f"A: {entry.get('answer', 'No answer')}\n"

    return transformed


def process_json_file(input_path, output_path):
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {input_path} is not a valid JSON file.")
        return
    except FileNotFoundError:
        print(f"Error: {input_path} not found.")
        return

    # Handle both list and dictionary formats
    if isinstance(data, dict):
        entries = [data]
    elif isinstance(data, list):
        entries = data
    else:
        print("Error: JSON data must be a list or dictionary.")
        return

    # Transform each entry
    transformed_entries = [transform_entry(entry) for entry in entries]

    # Write to output file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(transformed_entries))

    print(f"Successfully transformed {len(transformed_entries)} entries.")
    # print(f"Output written to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transform JSON data to context-Q&A format"
    )
    parser.add_argument("input_file", help="Path to input JSON file")
    parser.add_argument(
        "--output",
        "-o",
        default="transformed_output.txt",
        help="Path to output file (default: transformed_output.txt)",
    )

    args = parser.parse_args()

    process_json_file(args.input_file, args.output)
