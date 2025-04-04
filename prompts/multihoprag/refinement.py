import json
import re
from typing import Dict, List, Optional


def find_article_by_title(title: str, corpus: List[Dict]) -> Optional[Dict]:
    """Find an article in the corpus by its title."""
    # Clean the title for comparison
    clean_title = title.lower().strip()

    for article in corpus:
        article_title = article.get("title", "").lower().strip()
        if clean_title == article_title:
            print(f"Found article for title: {title}")
            return article

    print(f"Article not found for title: {title}")
    return None


def extract_paragraph_with_fact(article_body: str, fact_text: str) -> Optional[str]:
    """Extract the paragraph from the article body that contains the exact fact."""
    if not article_body or not fact_text:
        return None

    # Clean up fact text for comparison
    fact_text = fact_text.strip()

    # Split the article body into paragraphs
    paragraphs = re.split(r"\n+", article_body)

    # Look for paragraphs containing the exact fact text
    for para in paragraphs:
        para_text = para.strip()
        if fact_text in para_text:
            print("Found paragraph with the exact fact")
            return para_text

    print(f"Fact not found in article: {fact_text[:50]}...")
    return None


def process_article(title: str, fact: str, corpus: List[Dict]) -> Dict:
    """Process a single article to replace the fact with the original paragraph."""
    result = {"Title": title, "Fact": fact, "OriginalParagraph": None, "_meta": {}}

    if title and fact:
        # Find the article in the corpus
        article = find_article_by_title(title, corpus)

        if article:
            # Extract the original paragraph
            article_body = article.get("body", "")
            original_paragraph = extract_paragraph_with_fact(article_body, fact)

            if original_paragraph:
                # Save the original paragraph but don't replace the fact
                result["OriginalParagraph"] = original_paragraph
            else:
                # Note that the paragraph wasn't found
                result["_meta"] = {
                    "error": "Original paragraph not found in article body"
                }
        else:
            result["_meta"] = {"error": "Article not found in corpus"}

    return result


def main():
    # Configuration
    corpus_file = "corpus.json"
    input_file = "multihop_direct.txt"
    output_file = "output.txt"  # need to change

    # Load the corpus
    print("Loading corpus...")
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} articles in corpus")

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    raw_entries = re.split(r"\n{3,}", content)
    print(f"Found {len(raw_entries)} raw entries to process")

    # Process each entry
    with open(output_file, "w", encoding="utf-8") as f:
        for i, raw_entry in enumerate(raw_entries):
            print(f"Processing entry {i+1}/{len(raw_entries)}")

            try:
                # Split the entry into lines
                lines = raw_entry.strip().split("\n")

                # Initialize variables
                articles = []
                current_title = None
                current_fact = None
                question = None
                answer = None

                # State flags
                collecting_question = False
                collecting_answer = False

                # Process lines
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith("# Title:"):
                        # Save previous article if exists
                        if current_title and current_fact:
                            articles.append((current_title, current_fact))

                        # Start new article
                        current_title = line.replace("# Title:", "").strip()
                        current_fact = None
                    elif line.startswith("# Fact:"):
                        current_fact = line.replace("# Fact:", "").strip()
                    elif line.startswith("Q:"):
                        # Save the last article if necessary
                        if current_title and current_fact:
                            articles.append((current_title, current_fact))
                            current_title = None
                            current_fact = None

                        collecting_question = True
                        collecting_answer = False
                        question = line.replace("Q:", "").strip()
                    elif line.startswith("A:"):
                        collecting_question = False
                        collecting_answer = True
                        answer = line.replace("A:", "").strip()
                    elif collecting_question:
                        question = (question or "") + " " + line
                    elif collecting_answer:
                        answer = (answer or "") + " " + line

                # Save the last article if it exists
                if current_title and current_fact:
                    articles.append((current_title, current_fact))

                # Process all articles in this entry
                processed_articles = []

                for title, fact in articles:
                    result = process_article(title, fact, corpus)
                    processed_articles.append(result)
                entry_output = ""
                entry_output += "# Context\n"

                for article in processed_articles:
                    entry_output += f"# Article Title: {article['Title']}\n"
                    if article["OriginalParagraph"]:
                        entry_output += f"{article['OriginalParagraph']}\n\n"
                    else:
                        entry_output += "# Original paragraph not found\n\n"

                # Write Q&A
                if question:
                    entry_output += f"Q: {question}\n"
                if answer:
                    entry_output += f"A: {answer}\n"

                entry_output += "\n\n"

                # Write to file
                f.write(entry_output)

            except Exception as e:
                print(f"Error processing entry {i+1}: {e}")
                import traceback

                traceback.print_exc()

    print(f"Processing complete. Output written to {output_file}")


if __name__ == "__main__":
    main()
