# grab model checkpoint from huggingface hub

from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
from PIL import Image
import torch
import warnings
import argparse
import csv
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Suppress all warnings
warnings.filterwarnings("ignore")

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,
)

checkpoint_path = hf_hub_download(
    "openflamingo/OpenFlamingo-3B-vitl-mpt1b",
    "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)

model.eval()

# Here you can add other prompts that you want to test

PROMPT_LIST = [
    "<image>Classify this image into one of the following classes: comic, drawing, game, graph, logo, map, photo, title. "
]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate text for images in given classes.")
    parser.add_argument('--data_path', type=str, help='Path to the folder with classes subfolders')
    args = parser.parse_args()
    data_path = args.data_path

    if not os.path.exists('results'):
        os.makedirs('results')
    csv_file_path = os.path.join('results', 'results_flamingo.csv')

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['File Path', 'Prompt', 'Class Name', 'Predicted Text'])

        # Iterate over subfolders (classes)
        for class_name in os.listdir(data_path):
            class_path = os.path.join(data_path, class_name)
            print('Processing class: ', class_name)
            if os.path.isdir(class_path):
                for image_file in tqdm(os.listdir(class_path), total=len(os.listdir(class_path)), colour='red'):
                    file_path = os.path.join(class_path, image_file)
                    if os.path.isfile(file_path) and image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # Process each image
                        try:
                            query_image = Image.open(file_path)
                            vision_x = [image_processor(query_image).unsqueeze(0)]
                            vision_x = torch.cat(vision_x, dim=0)
                            vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(device)

                            for prompt in PROMPT_LIST:
                                lang_x = tokenizer(
                                    [prompt],
                                    return_tensors="pt",
                                )

                                generated_text = model.generate(
                                    vision_x=vision_x,
                                    lang_x=lang_x["input_ids"].to(device),
                                    attention_mask=lang_x["attention_mask"].to(device),
                                    max_new_tokens=20,
                                    num_beams=3,
                                    pad_token_id=50277
                                )

                                decoded_text = tokenizer.decode(generated_text[0])

                                # Remove the prompt from the generated text
                                decoded_text = decoded_text.replace(prompt, '').strip()
                                # Remove <|endofchunk|>
                                decoded_text = decoded_text.replace('<|endofchunk|>', '').strip()

                                writer.writerow([file_path, prompt, class_name, decoded_text])
                                file.flush()

                        except Exception as e:
                            print(f"Error processing file {file_path}: {e}")

    print('Done!')