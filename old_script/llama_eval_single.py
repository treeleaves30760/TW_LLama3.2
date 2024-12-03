import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate import Accelerator
from PIL import Image as PIL_Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from datetime import datetime
import json
import random
import math
import queue
import threading
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for generation (default: 0.7)')
    parser.add_argument('--top_p', type=float, default=0.9,
                      help='Top_p for generation (default: 0.9)')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model directory')
    parser.add_argument('--num_gpus', type=int, default=8,
                      help='Number of GPUs to use')
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

class GPUWorker(threading.Thread):
    def __init__(self, gpu_id, model_path, args, image_paths, result_queue):
        super().__init__()
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.args = args
        self.image_paths = image_paths  # List of images for this worker
        self.result_queue = result_queue
        self.device = f'cuda:{gpu_id}'
        
        # Create progress bar for this worker
        self.pbar = tqdm(
            total=len(image_paths),
            desc=f"GPU {gpu_id}",
            position=gpu_id,  # Position the progress bar
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
            
            # Process each image assigned to this worker
            for image_path in self.image_paths:
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
                        if location_name in response:
                            correct_count += 1
                        total_count += 1
                        
                        # Store result
                        results.append((str(image_path), question, response))
                    
                    # Update progress bar
                    self.pbar.update(1)
                    
                    # Clear CUDA cache after each image
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error processing image on GPU {self.gpu_id}: {str(e)}")
                    continue
            
            # Close progress bar
            self.pbar.close()
            
            # Send results back
            self.result_queue.put((results, correct_count, total_count))
            
        except Exception as e:
            print(f"Worker error on GPU {self.gpu_id}: {str(e)}")
            self.result_queue.put(([], 0, 0))
            self.pbar.close()

def create_markdown_report(all_results, model_name, output_dir):
    """Create evaluation report for the model."""
    time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    # Combine results from all GPUs
    combined_results = []
    total_correct = 0
    total_count = 0
    
    for results, correct, count in all_results:
        combined_results.extend(results)
        total_correct += correct
        total_count += count
    
    # Calculate overall accuracy
    accuracy = (total_correct / total_count * 100) if total_count > 0 else 0
    
    # Create report file
    report_path = output_dir / f'evaluation_{model_name}_{time_str}.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Evaluation Report for {model_name}\n\n")
        f.write(f"## Summary\n")
        f.write(f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Total Images: {total_count}\n")
        f.write(f"- Correct Predictions: {total_correct}\n")
        f.write(f"- Accuracy: {accuracy:.2f}%\n\n")
        
        f.write("## Detailed Results\n\n")
        for image_path, question, response in combined_results:
            rel_image_path = os.path.relpath(image_path, start=os.path.dirname(report_path))
            ground_truth = str(image_path).split('Images/')[1].split('-')[0]
            
            f.write(f"### Image: {os.path.basename(image_path)}\n")
            f.write(f"Ground Truth: {ground_truth}\n\n")
            f.write(f"![{os.path.basename(image_path)}]({rel_image_path})\n\n")
            f.write(f"Model Response: {response}\n")
            f.write("Correct: " + ("✓" if ground_truth in response else "✗") + "\n\n")
            f.write("---\n\n")

def main():
    # Initialize
    load_dotenv()
    args = parse_args()
    
    # Set up directory and get image files
    image_dir = Path(os.getenv('IMAGE_DIR'))
    image_files = list(image_dir.glob('*.[jJ][pP][gG]')) + \
                 list(image_dir.glob('*.[pP][nN][gG]')) + \
                 list(image_dir.glob('*.[jJ][pP][eE][gG]'))
    
    # Create output directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Set up result queue
    result_queue = queue.Queue()
    
    # Determine number of GPUs to use
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    
    # Distribute images among GPUs
    images_per_gpu = len(image_files) // num_gpus
    remainder = len(image_files) % num_gpus
    
    # Split image files among GPUs
    gpu_image_lists = []
    start_idx = 0
    for i in range(num_gpus):
        # Add one more image to some GPUs if there's a remainder
        current_chunk = images_per_gpu + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk
        gpu_image_lists.append(image_files[start_idx:end_idx])
        start_idx = end_idx
    
    print(f"\nStarting evaluation with {num_gpus} GPUs")
    print(f"Total images: {len(image_files)}")
    for i in range(num_gpus):
        print(f"GPU {i}: {len(gpu_image_lists[i])} images")
    print("\n")
    
    # Create and start GPU workers
    workers = []
    for gpu_id in range(num_gpus):
        worker = GPUWorker(
            gpu_id=gpu_id,
            model_path=args.model_path,
            args=args,
            image_paths=gpu_image_lists[gpu_id],
            result_queue=result_queue
        )
        worker.start()
        workers.append(worker)
    
    # Wait for all workers to complete
    for worker in workers:
        worker.join()
    
    # Move cursor past progress bars
    print("\n" * (num_gpus + 1))
    
    # Collect results
    all_results = []
    while not result_queue.empty():
        all_results.append(result_queue.get())
    
    # Generate report
    model_name = Path(args.model_path).name
    create_markdown_report(all_results, model_name, output_dir)
    print(f"Evaluation complete. Report saved in {output_dir}")

if __name__ == "__main__":
    main()