import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import Accelerator
from PIL import Image as PIL_Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from datetime import datetime
import json
import random
import math
import queue
import threading

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
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size for processing images per GPU')
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

def process_images_batch(image_paths: list) -> list:
    """Open and convert a batch of images from the specified paths."""
    images = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"The image file '{image_path}' does not exist.")
            continue
        try:
            with open(image_path, "rb") as f:
                images.append(PIL_Image.open(f).convert("RGB"))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
    return images

def generate_text_from_image_batch(model, processor, images, prompt_texts, temperature, top_p, device):
    """Generate text from a batch of images using the model and processor."""
    try:
        combined_prompts = []
        for prompt_text in prompt_texts:
            combined_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>你是一個旅遊專家，能十分準確的分析圖片中的景點。所有對話請用繁體中文進行，請嚴格按照使用者的提問進行回覆。<|end_of_text|>
<|start_header_id|>user<|end_header_id|>{prompt_text}<|image|><|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
            combined_prompts.append(combined_prompt)

        inputs = processor(
            text=combined_prompts,
            images=images,
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
        responses = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        return [response.strip() for response in responses]

    except Exception as e:
        print(f"Batch processing error on device {device}: {str(e)}")
        return ["Error processing batch"] * len(images)

class GPUWorker(threading.Thread):
    def __init__(self, gpu_id, model_path, args, task_queue, result_queue):
        super().__init__()
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.args = args
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.device = f'cuda:{gpu_id}'

    def run(self):
        try:
            # Set GPU device
            torch.cuda.set_device(self.gpu_id)
            
            # Load model and processor
            model, processor = load_model_and_processor(self.model_path, self.device)
            
            results = []
            correct_count = 0
            total_count = 0
            
            while True:
                try:
                    # Get batch of images from queue
                    batch = self.task_queue.get_nowait()
                    if batch is None:  # Sentinel value to stop the worker
                        break
                        
                    batch_files, location_names = batch
                    
                    # Process batch
                    images = process_images_batch([str(path) for path in batch_files])
                    questions = ["請問圖片中的景點是哪裡？景點有什麼特色？"] * len(batch_files)
                    
                    if images:  # Only process if we have valid images
                        responses = generate_text_from_image_batch(
                            model=model,
                            processor=processor,
                            images=images,
                            prompt_texts=questions,
                            temperature=self.args.temperature,
                            top_p=self.args.top_p,
                            device=self.device
                        )
                        
                        # Process results
                        for path, response, location_name in zip(batch_files, responses, location_names):
                            if location_name in response:
                                correct_count += 1
                            total_count += 1
                            results.append((str(path), questions[0], response))
                    
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    
                    # Update progress
                    self.task_queue.task_done()
                    
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"Error processing batch on GPU {self.gpu_id}: {str(e)}")
                    self.task_queue.task_done()
                    continue
            
            # Send results back
            self.result_queue.put((results, correct_count, total_count))
            
        except Exception as e:
            print(f"Worker error on GPU {self.gpu_id}: {str(e)}")
            self.result_queue.put(([], 0, 0))

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
    
    # Set up queues for task distribution
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # Create batches
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    batch_size = args.batch_size
    
    # Create all batches first
    all_batches = []
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        location_names = [str(path).split('Images/')[1].split('-')[0] for path in batch_files]
        all_batches.append((batch_files, location_names))
    
    # Put all batches in the task queue
    for batch in all_batches:
        task_queue.put(batch)
    
    # Add sentinel values to stop workers
    for _ in range(num_gpus):
        task_queue.put(None)
    
    # Create and start GPU workers
    workers = []
    for gpu_id in range(num_gpus):
        worker = GPUWorker(gpu_id, args.model_path, args, task_queue, result_queue)
        worker.start()
        workers.append(worker)
    
    # Wait for all workers to complete
    for worker in workers:
        worker.join()
    
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