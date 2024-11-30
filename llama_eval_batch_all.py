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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for generation (default: 0.7)')
    parser.add_argument('--top_p', type=float, default=0.9,
                      help='Top_p for generation (default: 0.9)')
    parser.add_argument('--models_dir', type=str, required=True,
                      help='Directory containing model folders')
    parser.add_argument('--num_gpus', type=int, default=8,
                      help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size for processing images')
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
    processor = MllamaProcessor.from_pretrained(model_path, use_safetensors=True)
    
    # 確保模型在正確的設備上並使用正確的數據類型
    model = model.to(device)
    model = model.to(torch.bfloat16)
    
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

def generate_text_from_image(
    model, processor, image, prompt_text: str, temperature: float, top_p: float, device
):
    """Generate text from an image using the model and processor."""
    combined_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>你是一個旅遊專家，能十分準確的分析圖片中的景點。所有對話請用繁體中文進行，請嚴格按照使用者的提問進行回覆。<|end_of_text|>
<|start_header_id|>user<|end_header_id|>{prompt_text}<|image|><|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    
    inputs = processor(
        text=combined_prompt,
        images=image,
        return_tensors="pt",
        add_special_tokens=False
    )
    
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in inputs.items()}
    
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=512,
            do_sample=True,
            use_cache=True,
            num_logits_to_keep=1
        )
    
    prompt_len = inputs['input_ids'].shape[1]
    generated_ids = output[:, prompt_len:]
    response = processor.decode(generated_ids[0], skip_special_tokens=True)
    
    return response.strip()

def generate_text_from_image_batch(
    model, processor, images, prompt_texts: list, temperature: float, top_p: float, device
):
    """Generate text from a batch of images using the model and processor."""
    combined_prompts = []
    for prompt_text in prompt_texts:
        combined_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>你是一個旅遊專家，能十分準確的分析圖片中的景點。所有對話請用繁體中文進行，請嚴格按照使用者的提問進行回覆。<|end_of_text|>
<|start_header_id|>user<|end_header_id|>{prompt_text}<|image|><|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
        combined_prompts.append(combined_prompt)
    
    try:
        inputs = processor(
            text=combined_prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        )
        
        # 確保所有張量使用相同的數據類型和設備
        inputs = {k: v.to(device).to(torch.bfloat16) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
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
        responses = [processor.decode(ids, skip_special_tokens=True).strip() 
                    for ids in generated_ids]
        
        return responses
        
    except RuntimeError as e:
        print(f"Error in batch processing: {str(e)}")
        # 如果批處理失敗，回退到單張處理
        return [generate_text_from_image(
            model, processor, img, prompt, temperature, top_p, device
        ) for img, prompt in zip(images, prompt_texts)]

def evaluate_model(model_path, image_files, args, device_id, TW_Attraction):
    """Evaluate a single model on all images using specified GPU."""
    device = f'cuda:{device_id}'
    model, processor = load_model_and_processor(model_path, device)
    
    results = []
    correct_count = 0
    total_count = 0
    
    # 將圖片分成批次
    batch_size = args.batch_size
    for i in tqdm(range(0, len(image_files), batch_size), 
                 desc=f"Processing {os.path.basename(model_path)} on GPU {device_id}"):
        batch_files = image_files[i:i + batch_size]
        location_names = [str(path).split('Images/')[1].split('-')[0] 
                         for path in batch_files]
        
        try:
            all_choice = ', '.join(TW_Attraction)
            questions = [f"請問圖片中的景點是哪裡？景點有什麼特色？"] * len(batch_files)
            
            images = process_images_batch([str(path) for path in batch_files])
            responses = generate_text_from_image_batch(
                model=model,
                processor=processor,
                images=images,
                prompt_texts=questions,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device
            )
            
            for path, question, response, location_name in zip(
                batch_files, questions, responses, location_names):
                if location_name in response:
                    correct_count += 1
                total_count += 1
                results.append((str(path), question, response))
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing batch with model {model_path}: {str(e)}")
            continue
            
    return results, correct_count, total_count

def create_markdown_report(all_results, output_dir):
    """Create individual markdown report for each model with model name in filename."""
    time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    # Create a summary file
    all_accuracies = {}
    
    # Process each model separately
    for model_name, (results, correct_count, total_count) in all_results.items():
        # Calculate accuracy
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        all_accuracies[model_name] = accuracy
        
        # Create individual report file with model name
        model_name_clean = model_name.replace('/', '_').replace('\\', '_')
        report_path = output_dir / f'evaluation_{model_name_clean}_{time_str}.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Write header with model details
            f.write(f"# Evaluation Report for {model_name}\n\n")
            f.write(f"## Summary\n")
            f.write(f"- Model: {model_name}\n")
            f.write(f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})\n\n")
            
            # Write evaluation details
            f.write("## Detailed Results\n\n")
            for image_path, question, response in results:
                rel_image_path = os.path.relpath(image_path, start=os.path.dirname(report_path))
                ground_truth = str(image_path).split('Images/')[1].split('-')[0]
                
                f.write(f"### Image: {os.path.basename(image_path)}\n")
                f.write(f"Ground Truth: {ground_truth}\n\n")
                f.write(f"![{os.path.basename(image_path)}]({rel_image_path})\n\n")
                f.write(f"Model Response: {response}\n")
                f.write("Correct: " + ("✓" if ground_truth in response else "✗") + "\n\n")
                f.write("---\n\n")
    
    # Create a summary file with all results
    summary_path = output_dir / f'summary_{time_str}.md'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Model Evaluation Summary\n\n")
        f.write("## Results Overview\n\n")
        f.write("| Model | Accuracy | Correct/Total |\n")
        f.write("|-------|-----------|---------------|\n")
        
        # Sort models by accuracy for easy comparison
        sorted_results = sorted(all_results.items(), 
                              key=lambda x: all_accuracies[x[0]], 
                              reverse=True)
        
        for model_name, (results, correct_count, total_count) in sorted_results:
            accuracy = all_accuracies[model_name]
            f.write(f"| {model_name} | {accuracy:.2f}% | {correct_count}/{total_count} |\n")

def main():
    # Load environment variables and parse arguments
    load_dotenv()
    args = parse_args()
    
    # Load TW_List
    with open('TW_List.json', 'r', encoding='utf-8') as f:
        TW_List = json.load(f)
        TW_Attraction = TW_List['TW_Attractions']
    
    # Get list of model directories
    models_dir = Path(args.models_dir)
    model_paths = [d for d in models_dir.iterdir() if d.is_dir()]
    
    # Get list of image files
    image_dir = Path(os.getenv('IMAGE_DIR'))
    image_files = list(image_dir.glob('*.[jJ][pP][gG]')) + \
                 list(image_dir.glob('*.[pP][nN][gG]')) + \
                 list(image_dir.glob('*.[jJ][pP][eE][gG]'))
    
    # Distribute models across available GPUs
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    models_per_gpu = len(model_paths) // num_gpus + (1 if len(model_paths) % num_gpus != 0 else 0)
    
    # Create output directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    all_results = {}
    
    # Process models in parallel across GPUs
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        future_to_model = {}
        
        for i, model_path in enumerate(model_paths):
            gpu_id = i % num_gpus
            future = executor.submit(
                evaluate_model,
                model_path,
                image_files,
                args,
                gpu_id,
                TW_Attraction
            )
            future_to_model[future] = model_path
        
        for future in tqdm(future_to_model, desc="Processing models"):
            model_path = future_to_model[future]
            try:
                results = future.result()
                all_results[model_path.name] = results
            except Exception as e:
                print(f"Error processing model {model_path}: {str(e)}")
    
    # Generate reports
    create_markdown_report(all_results, output_dir)
    print(f"Evaluation complete. Reports saved in {output_dir}")

if __name__ == "__main__":
    main()