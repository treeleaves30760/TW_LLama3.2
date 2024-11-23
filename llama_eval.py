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

# Initialize accelerator
accelerator = Accelerator()
device = accelerator.device

# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['1epoch', '3epochs', '5epochs'],
                      help='Select model version', default='5epochs')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for generation (default: 0.7)')
    parser.add_argument('--top_p', type=float, default=0.9,
                      help='Top p for generation (default: 0.9)')
    return parser.parse_args()

def load_model_and_processor(model_version):
    """Load the specified version of the fine-tuned model."""
    base_model_path = os.getenv('LLAMA_BASE')
    model_paths = {
        '1epoch': os.getenv('LLAMA_1EPOCH'),
        '3epochs': os.getenv('LLAMA_3EPOCHS'),
        '5epochs': os.getenv('LLAMA_5EPOCHS')
    }
    
    model_path = model_paths[model_version]
    print(f"Loading model from {model_path}...")
    
    # Load model and processor
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map=device,
    )
    processor = MllamaProcessor.from_pretrained(model_path, use_safetensors=True)

    # Prepare model and processor with accelerator
    model, processor = accelerator.prepare(model, processor)
    base_model = MllamaForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map=device,
    )
    return model, base_model, processor

def process_image(image_path: str) -> PIL_Image.Image:
    """Open and convert an image from the specified path."""
    if not os.path.exists(image_path):
        print(f"The image file '{image_path}' does not exist.")
        sys.exit(1)
    with open(image_path, "rb") as f:
        return PIL_Image.open(f).convert("RGB")

def generate_text_from_image(
    model, processor, image, prompt_text: str, temperature: float, top_p: float
):
    """Generate text from an image using the model and processor."""
    # Combine system prompt and user prompt with image token
    combined_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>你是一個旅遊專家，能十分準確的分析圖片中的景點。分析景點時請先列出這個照片的細節，再推測這是那個景點。所有對話請用繁體中文進行。<|end_of_text|>
<|start_header_id|>user<|end_header_id|><|image|>{prompt_text}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    with_location_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>你是一個旅遊專家，能十分準確的分析圖片中的景點。分析景點時請先列出這個照片的細節，再推測這是那個景點。所有對話請用繁體中文進行。<|end_of_text|>
<|start_header_id|>user<|end_header_id|><|image|>{prompt_text}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    
    # Process inputs using the processor
    inputs = processor(
        text=combined_prompt,
        images=image,
        return_tensors="pt",
        add_special_tokens=False
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in inputs.items()}
    
    # Generate response
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=512,
            do_sample=True,
            use_cache=True,
            num_logits_to_keep=1  # Only keep logits for the last token to save memory
        )
    
    # Get the generated text after the prompt
    prompt_len = inputs['input_ids'].shape[1]
    generated_ids = output[:, prompt_len:]
    response = processor.decode(generated_ids[0], skip_special_tokens=True)
    
    return response.strip()

def create_markdown_report(results, output_path):
    """Create a markdown report with images and responses."""
    markdown_content = "# LLaMA Vision Model Evaluation Report\n\n"
    
    for image_path, response, base_response, with_location_response in results:
        rel_image_path = os.path.relpath(image_path, start=os.path.dirname(output_path))
        markdown_content += f"## Image: {os.path.basename(image_path)}\n\n"
        markdown_content += f"![{os.path.basename(image_path)}]({rel_image_path})\n\n"
        markdown_content += "### Model Response:\n\n"
        markdown_content += f"{response}\n\n"
        markdown_content += "### Base Model Response:\n\n"
        markdown_content += f"{base_response}\n\n"
        markdown_content += "### Model Response with Location:\n\n"
        markdown_content += f"{with_location_response}\n\n"
        markdown_content += "---\n\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

def main():
    args = parse_args()
    
    print(f"Loading {args.model} model...")
    model, base_model, processor = load_model_and_processor(args.model)
    
    # Get list of image files
    image_dir = Path(os.getenv('IMAGE_DIR'))
    image_files = list(image_dir.glob('*.[jJ][pP][gG]')) + \
                 list(image_dir.glob('*.[pP][nN][gG]')) + \
                 list(image_dir.glob('*.[jJ][pP][eE][gG]'))
    
    # Process each image
    results = []
    question = "請問圖片中的景點是哪裡？景點有什麼特色？"
    
    print("Processing images...")
    for image_path in tqdm(image_files):
        try:
            # Process one image at a time to manage memory
            location_name = str(image_path).split('images/')[1].split('-')[0]
            question_with_location = f"圖片中的景點是{location_name}, 請告訴我關於這個景點的事情"
            image = process_image(str(image_path))
            response = generate_text_from_image(
                model=model, 
                processor=processor, 
                image=image, 
                prompt_text=question,
                temperature=args.temperature,
                top_p=args.top_p
            )
            base_response = generate_text_from_image(
                model=base_model, 
                processor=processor, 
                image=image, 
                prompt_text=question,
                temperature=args.temperature,
                top_p=args.top_p
            )
            with_location_response = generate_text_from_image(
                model=model, 
                processor=processor, 
                image=image, 
                prompt_text=question_with_location,
                temperature=args.temperature,
                top_p=args.top_p
            )
            results.append((str(image_path), response, base_response, with_location_response))
            
            # Clear cache after each image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    # Create output directory if it doesn't exist
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Generate report
    time_str = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    output_path = output_dir / f'evaluation_report_{args.model}_{time_str}.md'
    create_markdown_report(results, output_path)
    print(f"Evaluation complete. Report saved to {output_path}")

if __name__ == "__main__":
    main()