#!/usr/bin/env python3
import json
import csv
import argparse

def convert_tsv_to_json(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t')
            headers = next(reader)
            
            json_data = []
            for row in reader:
                entry = {}
                
                # Map each value to its corresponding header
                for i, header in enumerate(headers):
                    if i < len(row):
                        entry[header] = row[i]
                    else:
                        entry[header] = ""  # Or None, depending on your preference
                
                json_data.append(entry)
        
        # Write the JSON to the output file
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=4)
        
        print(f"Successfully converted {input_file} to {output_file}")
        return json_data
    
    except Exception as e:
        print(f"Error converting TSV to JSON: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert TSV file to JSON')
    parser.add_argument('input_file', help='Path to the input TSV file')
    parser.add_argument('output_file', help='Path to save the output JSON file')
    
    args = parser.parse_args()
    
    convert_tsv_to_json(args.input_file, args.output_file)
