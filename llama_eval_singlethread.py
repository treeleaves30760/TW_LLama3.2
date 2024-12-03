import argparse
import os
import sys
from pathlib import Path
import torch
import threading
from datetime import datetime
from PIL import Image as PIL_Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--folder_path', type=str, required=True,
                      help='Path to the specific data folder to process')
    parser.add_argument('--gpu_id', type=int, required=True,
                      help='GPU ID to use')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to save results JSON')
    return parser.parse_args()

def load_model_and_processor(model_path, device):
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

def evaluate_folder(args):
    # Set GPU device
    device = f'cuda:{args.gpu_id}'
    torch.cuda.set_device(args.gpu_id)
    
    # Load model and processor
    model, processor = load_model_and_processor(args.model_path, device)
    
    # Get all image files
    folder_path = Path(args.folder_path)
    image_files = list(folder_path.glob('*.[jJ][pP][gG]')) + \
                 list(folder_path.glob('*.[pP][nN][gG]')) + \
                 list(folder_path.glob('*.[jJ][pP][eE][gG]'))
    
    results = []
    correct_count = 0
    total_count = 0
    
    # Create progress bar
    pbar = tqdm(total=len(image_files), desc=f"GPU {args.gpu_id} - {folder_path.name}")
    
    for image_path in image_files:
        try:
            image = process_single_image(str(image_path))
            if image:
                question = "請問圖片中的景點是哪裡？景點有什麼特色？"
                response = generate_text_from_image(
                    model=model,
                    processor=processor,
                    image=image,
                    prompt_text=question,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device
                )
                
                # Get ground truth from image path
                # File name: ~/TW_LLama3.2/Distributed_Images/folder_1/921地震教育園區-1.jpg
                # Location: 921地震教育園區
                location_name = str(image_path).split('Distributed_Images/folder_')[1][2:].split('-')[0]
                
                # Check if correct
                is_correct = location_name in response
                if is_correct:
                    correct_count += 1
                total_count += 1
                
                # Store result
                results.append({
                    'image_path': str(image_path),
                    'ground_truth': location_name,
                    'response': response,
                    'is_correct': is_correct
                })
            
            pbar.update(1)
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing image on GPU {args.gpu_id}: {str(e)}")
            continue
    
    pbar.close()
    
    # Calculate accuracy
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    # Prepare final results
    final_results = {
        'folder': folder_path.name,
        'gpu_id': args.gpu_id,
        'results': results,
        'correct_count': correct_count,
        'total_count': total_count,
        'accuracy': accuracy,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save results
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation complete for folder {folder_path.name}. Results saved to {args.output_path}")

def main():
    args = parse_args()
    evaluate_folder(args)

if __name__ == "__main__":
    main()