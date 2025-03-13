import re
import json
import argparse
from typing import List, Dict, Any

def parse_flan_examples(input_file: str) -> List[Dict[str, Any]]:
    """Parse examples from a file in FLAN-T5 format."""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by metadata markers
    examples = []
    parts = re.split(r'# METADATA: ', content)
    
    # Skip the first part if it's empty
    parts = [p for p in parts if p.strip()]
    
    for part in parts:
        # Extract metadata
        metadata_match = re.match(r'(\{.*?\})(.*)', part, re.DOTALL)
        if not metadata_match:
            continue
            
        metadata_str, example_text = metadata_match.groups()
        try:
            metadata = json.loads(metadata_str)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse metadata: {metadata_str}")
            continue
        
        # Extract Wikipedia contexts
        contexts = []
        wiki_pattern = r'Wikipedia Title: (.*?)\n(.*?)(?=Wikipedia Title:|Q:|$)'
        for match in re.finditer(wiki_pattern, example_text, re.DOTALL):
            title = match.group(1).strip()
            content = match.group(2).strip()
            contexts.append({"title": title, "content": content})
        
        # Extract question and answer
        question_match = re.search(r'Q:\s*(.*?)(?=\nA:|\Z)', example_text, re.DOTALL)
        answer_match = re.search(r'A:\s*(.*?)(?=\Z|\n# METADATA:)', example_text, re.DOTALL)
        
        if not question_match:
            print(f"Warning: No question found for example with metadata {metadata}")
            continue
            
        question = question_match.group(1).strip()
        answer = answer_match.group(1).strip() if answer_match else ""
        
        examples.append({
            "metadata": metadata,
            "contexts": contexts,
            "question": question,
            "answer": answer
        })
    
    return examples

def format_as_llama_instruct(
    examples: List[Dict[str, Any]], 
    system_prompt: str,
    direct_answer: bool = True
) -> str:
    """Convert parsed examples to Llama-instruct format."""
    result = []
    
    for example in examples:
        # Format metadata as a comment
        formatted_example = f"# METADATA: {json.dumps(example['metadata'])}\n"
        
        # Start Llama format
        formatted_example += "<s>[INST] <<SYS>>\n"
        formatted_example += f"{system_prompt}\n"
        formatted_example += "<</SYS>>\n\n"
        
        # Format context
        formatted_example += "# Context\n"
        for ctx in example["contexts"]:
            formatted_example += f"Wikipedia Title: {ctx['title']}\n{ctx['content']}\n\n"
        
        # Format question
        formatted_example += "# Question\n"
        formatted_example += f"{example['question']} [/INST]\n\n"
        
        # Format answer
        formatted_example += f"{example['answer']} </s>"
        
        result.append(formatted_example)
    
    return "\n\n".join(result)

def main():
    parser = argparse.ArgumentParser(description="Convert FLAN-T5 format to Llama-instruct format")
    parser.add_argument("--input", "-i", required=True, help="Input file in FLAN-T5 format")
    parser.add_argument("--output", "-o", required=True, help="Output file for Llama-instruct format")
    parser.add_argument("--system-prompt", "-s", default="You are a helpful AI assistant. Answer the question accurately based on the provided context.", 
                       help="System prompt to use")
    parser.add_argument("--direct", "-d", action="store_true", 
                       help="Format for direct answers (no reasoning)")
    args = parser.parse_args()
    
    # Parse examples from input file
    examples = parse_flan_examples(args.input)
    print(f"Parsed {len(examples)} examples from {args.input}")
    
    # Convert to Llama-instruct format
    formatted_content = format_as_llama_instruct(
        examples, 
        args.system_prompt,
        direct_answer=args.direct
    )
    
    # Write to output file
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(formatted_content)
    
    print(f"Converted examples written to {args.output}")

if __name__ == "__main__":
    main()