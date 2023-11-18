import cv2
import torch
from PIL import Image

import llama
import os
import argparse
import csv
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_dir = "/scratch/language-models-resources/llama/"

model, preprocess = llama.load(name="BIAS-7B",
                               llama_dir=llama_dir,
                               llama_type="7B",
                               device=device)
model.half()
model.eval()

def multi_modal_generate(
    img_path: str,
    prompt: str,
    max_gen_len=256,
    temperature: float = 0.1,
    top_p: float = 0.75,
):
    try:
        img = Image.fromarray(cv2.imread(img_path))
    except:
        return ""

    img = preprocess(img).unsqueeze(0).half().to(device)
    prompt = llama.format_prompt(prompt)

    result = model.generate(img, [prompt],
                            max_gen_len=max_gen_len,
                            temperature=temperature,
                            top_p=top_p)
    print(result[0])
    return result[0]

PROMPT_LIST = [
    "Classify this image into one of the following classes: comic, drawing, game, graph, logo, map, photo, title. "
]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate text for images in given classes.")
    parser.add_argument('--data_path', type=str, help='Path to the folder with classes subfolders')
    args = parser.parse_args()
    data_path = args.data_path

    if not os.path.exists('results'):
        os.makedirs('results')
    csv_file_path = os.path.join('results', 'results_llama.csv')

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
                        try:
                            for prompt in PROMPT_LIST:
                                decoded_text = multi_modal_generate(file_path, prompt,
                                                                    max_gen_len=64,
                                                                    temperature=0.1,
                                                                    top_p=0.75)
                                writer.writerow([file_path, prompt, class_name, decoded_text])
                                file.flush()
                        except Exception as e:
                            print(f"Error processing file {file_path}: {e}")

    print('Done!')