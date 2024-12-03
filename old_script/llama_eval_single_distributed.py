import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import torch
import threading
import queue
from datetime import datetime
from PIL import Image as PIL_Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for generation (default: 0.7)')
    parser.add_argument('--top_p', type=float, default=0.9,
                      help='Top_p for generation (default: 0.9)')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model directory')
    parser.add_argument('--data_root', type=str, required=True,
                      help='Root directory containing the 8 data folders')
    parser.add_argument('--config_output', type=str, default='config_results.json',
                      help='Path to save configuration results')
    return parser.parse_args()

def load_model_and_processor(model_path, device):
    """Load model and processor on specified device."""
    print(f"Loading model from {model_path} on device {device}...")
    
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map=device,
    )
    processor = MllamaProcessor.from_pretrained(
        model_path, 
        use_safetensors=True,
        padding_side='left'
    )
    
    model = model.eval()
    model = model.to(device)
    
    return model, processor

def process_single_image(image_path: str) -> PIL_Image.Image:
    """Open and convert a single image from the specified path."""
    if not os.path.exists(image_path):
        print(f"The image file '{image_path}' does not exist.")
        return None
    try:
        with open(image_path, "rb") as f:
            return PIL_Image.open(f).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def generate_text_from_image(model, processor, image, prompt_text, temperature, top_p, device):
    """Generate text from a single image using the model and processor."""
    try:
        combined_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>你是一個旅遊專家，能十分準確的分析圖片中的景點。所有對話請用繁體中文進行，請嚴格按照使用者的提問進行回覆。<|end_of_text|>
<|start_header_id|>user<|end_header_id|>{prompt_text}<|image|><|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

        inputs = processor(
            text=[combined_prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        )
        
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=512,
                do_sample=True,
                use_cache=True,
                num_logits_to_keep=1
            )

        prompt_len = inputs['input_ids'].shape[1]
        generated_ids = outputs[:, prompt_len:]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response.strip()

    except Exception as e:
        print(f"Image processing error on device {device}: {str(e)}")
        return "Error processing image"

class FolderWorker(threading.Thread):
    def __init__(self, gpu_id, folder_path, model_path, args, result_queue, lock):
        super().__init__()
        self.gpu_id = gpu_id
        self.folder_path = folder_path
        self.model_path = model_path
        self.args = args
        self.result_queue = result_queue
        self.lock = lock
        self.device = f'cuda:{gpu_id}'
        
        # Get all image files in the folder
        self.image_files = list(Path(folder_path).glob('*.[jJ][pP][gG]')) + \
                          list(Path(folder_path).glob('*.[pP][nN][gG]')) + \
                          list(Path(folder_path).glob('*.[jJ][pP][eE][gG]'))
        
        # Create progress bar
        self.pbar = tqdm(
            total=len(self.image_files),
            desc=f"GPU {gpu_id} - {Path(folder_path).name}",
            position=gpu_id,
            leave=True
        )

    def run(self):
        try:
            # Set GPU device
            torch.cuda.set_device(self.gpu_id)
            
            # Load model and processor
            model, processor = load_model_and_processor(self.model_path, self.device)
            
            results = []
            correct_count = 0
            total_count = 0
            
            # Create individual markdown file for this folder
            folder_name = Path(self.folder_path).name
            markdown_path = Path('results') / f'evaluation_{folder_name}.md'
            
            with open(markdown_path, 'w', encoding='utf-8') as md_file:
                md_file.write(f"# Evaluation Results for {folder_name}\n\n")
                
                # Process each image
                for image_path in self.image_files:
                    try:
                        image = process_single_image(str(image_path))
                        if image:
                            question = "請問圖片中的景點是哪裡？景點有什麼特色？"
                            response = generate_text_from_image(
                                model=model,
                                processor=processor,
                                image=image,
                                prompt_text=question,
                                temperature=self.args.temperature,
                                top_p=self.args.top_p,
                                device=self.device
                            )
                            
                            # Get ground truth from image path
                            location_name = str(image_path).split('Images/')[1].split('-')[0]
                            
                            # Check if correct
                            is_correct = location_name in response
                            if is_correct:
                                correct_count += 1
                            total_count += 1
                            
                            # Write to markdown file
                            with self.lock:
                                md_file.write(f"### Image: {image_path.name}\n")
                                md_file.write(f"Ground Truth: {location_name}\n\n")
                                md_file.write(f"![{image_path.name}]({image_path})\n\n")
                                md_file.write(f"Model Response: {response}\n")
                                md_file.write("Correct: " + ("✓" if is_correct else "✗") + "\n\n")
                                md_file.write("---\n\n")
                            
                            # Store result
                            results.append({
                                'image_path': str(image_path),
                                'ground_truth': location_name,
                                'response': response,
                                'is_correct': is_correct
                            })
                        
                        # Update progress bar
                        self.pbar.update(1)
                        
                        # Clear CUDA cache
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"Error processing image on GPU {self.gpu_id}: {str(e)}")
                        continue
            
            # Close progress bar
            self.pbar.close()
            
            # Calculate accuracy for this folder
            accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
            
            # Send results back
            self.result_queue.put({
                'folder': folder_name,
                'results': results,
                'correct_count': correct_count,
                'total_count': total_count,
                'accuracy': accuracy
            })
            
        except Exception as e:
            print(f"Worker error on GPU {self.gpu_id}: {str(e)}")
            self.result_queue.put({
                'folder': Path(self.folder_path).name,
                'results': [],
                'correct_count': 0,
                'total_count': 0,
                'accuracy': 0
            })
            self.pbar.close()

def create_final_report(all_results, model_name):
    """Create final evaluation report combining results from all folders."""
    time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    report_path = Path('results') / f'final_evaluation_{model_name}_{time_str}.md'
    
    total_correct = sum(r['correct_count'] for r in all_results)
    total_count = sum(r['total_count'] for r in all_results)
    overall_accuracy = (total_correct / total_count * 100) if total_count > 0 else 0
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Final Evaluation Report for {model_name}\n\n")
        f.write(f"## Overall Summary\n")
        f.write(f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Total Images: {total_count}\n")
        f.write(f"- Correct Predictions: {total_correct}\n")
        f.write(f"- Overall Accuracy: {overall_accuracy:.2f}%\n\n")
        
        f.write("## Results by Folder\n\n")
        for result in all_results:
            f.write(f"### {result['folder']}\n")
            f.write(f"- Total Images: {result['total_count']}\n")
            f.write(f"- Correct Predictions: {result['correct_count']}\n")
            f.write(f"- Accuracy: {result['accuracy']:.2f}%\n\n")

def save_config_results(all_results, config_path):
    """Save configuration results to JSON file."""
    config_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'overall_results': {
            'total_images': sum(r['total_count'] for r in all_results),
            'total_correct': sum(r['correct_count'] for r in all_results),
            'overall_accuracy': (sum(r['correct_count'] for r in all_results) / 
                               sum(r['total_count'] for r in all_results) * 100)
                               if sum(r['total_count'] for r in all_results) > 0 else 0
        },
        'folder_results': {
            r['folder']: {
                'total_images': r['total_count'],
                'correct_predictions': r['correct_count'],
                'accuracy': r['accuracy']
            }
            for r in all_results
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_results, f, ensure_ascii=False, indent=2)

def main():
    # Initialize
    load_dotenv()
    args = parse_args()
    
    # Create output directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Get all folder paths
    data_root = Path(args.data_root)
    folder_paths = [f for f in data_root.iterdir() if f.is_dir()]
    
    if len(folder_paths) != 8:
        raise ValueError(f"Expected 8 folders in {data_root}, found {len(folder_paths)}")
    
    # Set up result queue and thread lock
    result_queue = queue.Queue()
    lock = threading.Lock()
    
    print(f"\nStarting evaluation with 8 folders")
    print("Folder distribution:")
    for i, folder in enumerate(folder_paths):
        print(f"GPU {i}: {folder.name}")
    print("\n")
    
    # Create and start workers
    workers = []
    for gpu_id, folder_path in enumerate(folder_paths):
        worker = FolderWorker(
            gpu_id=gpu_id,
            folder_path=folder_path,
            model_path=args.model_path,
            args=args,
            result_queue=result_queue,
            lock=lock
        )
        worker.start()
        workers.append(worker)
    
    # Wait for all workers to complete
    for worker in workers:
        worker.join()
    
    # Move cursor past progress bars
    print("\n" * (len(workers) + 1))
    
    # Collect results
    all_results = []
    while not result_queue.empty():
        all_results.append(result_queue.get())
    
    # Sort results by folder name
    all_results.sort(key=lambda x: x['folder'])
    
    # Generate final report
    model_name = Path(args.model_path).name
    create_final_report(all_results, model_name)
    
    # Save configuration results
    save_config_results(all_results, args.config_output)
    
    print(f"Evaluation complete. Results saved in {output_dir}")
    print(f"Configuration results saved to {args.config_output}")

if __name__ == "__main__":
    main()