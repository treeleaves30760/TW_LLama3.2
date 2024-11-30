import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import torch
from accelerate import Accelerator
from PIL import Image as PIL_Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from datetime import datetime
import json
import random
from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp
import torch.distributed as dist
import math

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
        # torch_dtype=torch.bfloat16,
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
        with open(image_path, "rb") as f:
            images.append(PIL_Image.open(f).convert("RGB"))
    return images

def generate_text_from_image_batch(
    model, processor, images, prompt_texts: list, temperature: float, top_p: float, device
):
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
        
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=512,
                do_sample=True,
                use_cache=True,
                num_logits_to_keep=1,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        prompt_len = inputs['input_ids'].shape[1]
        generated_ids = outputs[:, prompt_len:]
        responses = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        return [response.strip() for response in responses]
        
    except Exception as e:
        print(f"Batch processing error: {str(e)}")
        return ["Error processing batch"] * len(images)

def evaluate_model_shard(args, image_files, device_id, start_idx, end_idx):
    """Evaluate a portion of images on a specific GPU."""
    device = f'cuda:{device_id}'
    torch.cuda.set_device(device)
    
    model, processor = load_model_and_processor(args.model_path, device)
    
    results = []
    correct_count = 0
    total_count = 0
    
    # Get the shard of images for this GPU
    shard_files = image_files[start_idx:end_idx]
    
    # Create batches for this shard
    batches = [(shard_files[i:i + args.batch_size], 
                [str(path).split('Images/')[1].split('-')[0] for path in shard_files[i:i + args.batch_size]])
               for i in range(0, len(shard_files), args.batch_size)]
    
    for batch_files, location_names in tqdm(batches, 
                                          desc=f"Processing on GPU {device_id} ({start_idx}:{end_idx})"):
        try:
            images = process_images_batch([str(path) for path in batch_files])
            questions = ["請問圖片中的景點是哪裡？景點有什麼特色？"] * len(batch_files)
            
            responses = generate_text_from_image_batch(
                model=model,
                processor=processor,
                images=images,
                prompt_texts=questions,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device
            )
            
            for path, response, location_name in zip(batch_files, responses, location_names):
                if location_name in response:
                    correct_count += 1
                total_count += 1
                results.append((str(path), questions[0], response))
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing batch on GPU {device_id}: {str(e)}")
            continue
            
    return results, correct_count, total_count

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
        f.write(f"- Accuracy: {accuracy:.2f}% ({total_correct}/{total_count})\n\n")
        
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
    load_dotenv()
    args = parse_args()
    
    # Get list of image files
    image_dir = Path(os.getenv('IMAGE_DIR'))
    image_files = list(image_dir.glob('*.[jJ][pP][gG]')) + \
                 list(image_dir.glob('*.[pP][nN][gG]')) + \
                 list(image_dir.glob('*.[jJ][pP][eE][gG]'))
    
    # Calculate images per GPU
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    images_per_gpu = math.ceil(len(image_files) / num_gpus)
    
    # Create output directory
    output_dir = Path('results', str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
    output_dir.mkdir(exist_ok=True)
    
    # Process images in parallel across GPUs
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        
        for gpu_id in range(num_gpus):
            start_idx = gpu_id * images_per_gpu
            end_idx = min((gpu_id + 1) * images_per_gpu, len(image_files))
            
            future = executor.submit(
                evaluate_model_shard,
                args,
                image_files,
                gpu_id,
                start_idx,
                end_idx
            )
            futures.append(future)
        
        # Collect results
        all_results = []
        for future in tqdm(futures, desc="Collecting results from GPUs"):
            try:
                results = future.result()
                all_results.append(results)
            except Exception as e:
                print(f"Error collecting results: {str(e)}")
    
    # Generate report
    model_name = Path(args.model_path).name
    create_markdown_report(all_results, model_name, output_dir)
    print(f"Evaluation complete. Report saved in {output_dir}")

if __name__ == "__main__":
    main()