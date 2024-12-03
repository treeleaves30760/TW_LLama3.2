import argparse
import json
from pathlib import Path
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                      help='Directory containing individual JSON result files')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name of the model being evaluated')
    parser.add_argument('--output_dir', type=str, default='final_results',
                      help='Directory to save final results')
    return parser.parse_args()

def create_markdown_report(all_results, model_name, output_path):
    """Create detailed markdown report."""
    total_correct = sum(r['correct_count'] for r in all_results)
    total_count = sum(r['total_count'] for r in all_results)
    overall_accuracy = (total_correct / total_count * 100) if total_count > 0 else 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Final Evaluation Report for {model_name}\n\n")
        f.write(f"## Overall Summary\n")
        f.write(f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Total Images: {total_count}\n")
        f.write(f"- Correct Predictions: {total_correct}\n")
        f.write(f"- Overall Accuracy: {overall_accuracy:.2f}%\n\n")
        
        f.write("## Results by Folder\n\n")
        for result in sorted(all_results, key=lambda x: x['folder']):
            f.write(f"### {result['folder']}\n")
            f.write(f"- GPU ID: {result['gpu_id']}\n")
            f.write(f"- Total Images: {result['total_count']}\n")
            f.write(f"- Correct Predictions: {result['correct_count']}\n")
            f.write(f"- Accuracy: {result['accuracy']:.2f}%\n")
            f.write(f"- Timestamp: {result['timestamp']}\n\n")

def create_json_report(all_results, model_name, output_path):
    """Create detailed JSON report."""
    total_correct = sum(r['correct_count'] for r in all_results)
    total_count = sum(r['total_count'] for r in all_results)
    overall_accuracy = (total_correct / total_count * 100) if total_count > 0 else 0
    
    final_results = {
        'model_name': model_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'overall_results': {
            'total_images': total_count,
            'total_correct': total_correct,
            'overall_accuracy': overall_accuracy
        },
        'folder_results': {
            r['folder']: {
                'gpu_id': r['gpu_id'],
                'total_images': r['total_count'],
                'correct_predictions': r['correct_count'],
                'accuracy': r['accuracy'],
                'timestamp': r['timestamp']
            }
            for r in all_results
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load all result files
    results_dir = Path(args.results_dir)
    result_files = list(results_dir.glob('*.json'))
    
    if not result_files:
        print(f"No JSON result files found in {results_dir}")
        return
    
    # Load and combine results
    all_results = []
    for result_file in result_files:
        with open(result_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
            all_results.append(result_data)
    
    # Generate reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    markdown_path = output_dir / f'final_report_{args.model_name}_{timestamp}.md'
    create_markdown_report(all_results, args.model_name, markdown_path)
    
    json_path = output_dir / f'final_report_{args.model_name}_{timestamp}.json'
    create_json_report(all_results, args.model_name, json_path)
    
    print(f"Final reports generated:")
    print(f"- Markdown report: {markdown_path}")
    print(f"- JSON report: {json_path}")

if __name__ == "__main__":
    main()