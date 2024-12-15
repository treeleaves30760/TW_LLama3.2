import json

def process_json_file(json_file):
    # Read the JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each result
    output_rows = []
    for result in data['results']:
        # Clean up the response by replacing newlines with space
        cleaned_response = result['response'].replace('\n', ' ')
        
        # Create a row with tab separation (for easy spreadsheet pasting)
        row = f"{result['image_path'].split('folder_')[1][2:]}\t{cleaned_response}\t{result['is_correct']}"
        output_rows.append(row)
    
    # Join all rows with newlines
    output_text = '\n'.join(output_rows)
    
    # Save the output text to a file
    output_file = json_file.replace('.json', '_formatted.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_text)

# Usage example
if __name__ == "__main__":
    input_file = "/media/Pluto/stanley_hsu/github/TW_LLama3.2/results/evaluation_results_20241211_220456/folder_8_results.json"
    process_json_file(input_file)