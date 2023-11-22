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

possible_labels = [
    "drawing",
    "game",
    "graph",
    "logo",
    "comic",
    "map",
    "photo",
    "title",
]

image_file_paths = []
for root, dirs, files in os.walk("data/test"):
    for file in files:
        if file.lower().endswith(".jpg"):
            image_file_paths.append(os.path.join(root, file))

images = []
for image_path in image_file_paths:
    image = Image.open(image_path)
    images.append(image)

labelled_data = []
for image_path in image_file_paths:
    label = os.path.basename(os.path.dirname(image_path))
    if label in possible_labels:
        img_data = {
            "img": Image.open(image_path),
            "label": label,
            "label_id": possible_labels.index(label),
            "path": image_path
        }
        labelled_data.append(img_data)

true_labels = [item["label_id"] for item in labelled_data]
img_paths = [item["path"] for item in labelled_data]


# Here you can add other prompts that you want to test
#PROMPT_LIST = [
    #"<image>Classify this image into one of the following classes: comic, drawing, game, graph, logo, map, photo, title. ",
    #"<image>Is this a comic or a drawing or a game or a graph or a logo or a map or a photo or a title? ",
    #"<image>Classify this image. ",
    #"<image>What class of image is this? ",
    #"<image>How would you classify this image? ",
    #"<image>What type of image is this? ",
    #"<image>Describe this image. ",
    #"<image>Describe the content, structure, visual features, textual information and context of this image. ",
    #"<image>Caption this image. ",
    #"<image>An image of",
    #"<image>This image can be classified as a",
    #"<image> Between the classes comic, drawing, game, graph, logo, map, photograph and title, the correct class for this image is",
#]
prompt = "<image> Between the classes comic, drawing, game, graph, logo, map, photograph and title,this image is a"

def ask_flamingo(img_path, prompt): 
    try: 
        query_image = Image.open(img_path)
        vision_x = [image_processor(query_image).unsqueeze(0)]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(device)
        
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

    except Exception as e:
                            print(f"Error processing file {img_path}: {e}")
    return decoded_text


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate text for images in given classes.")
    parser.add_argument('--data_path', type=str, help='Path to the folder with classes subfolders')
    args = parser.parse_args()
    data_path = args.data_path

    folder_index = str(len(os.listdir("./results")) + 1)
    folder_path = os.path.join("./results", "results" + folder_index)
    #CSV for the results of this prompt
    csv_name = "results" + folder_index + ".csv"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    csv_file_path = os.path.join(folder_path, csv_name)
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['File Path', 'Prompt', 'Class Name', 'Predicted Text'])

        # Iterate over subfolders (classes)
        for class_name in os.listdir(data_path):
            class_path = os.path.join(data_path, class_name)
            print('Processing class: ', class_name)
            if os.path.isdir(class_path):
                for image_file in tqdm(os.listdir(class_path), total=len(os.listdir(class_path)), colour='red'):
                    img_path = os.path.join(class_path, image_file)
                    if os.path.isfile(img_path) and image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # Process each image
                        try:
                            decoded_text = ask_flamingo(img_path, prompt)
                            print(f"Decoded text: {decoded_text}")
                            writer.writerow([img_path, prompt, class_name, decoded_text])
                            file.flush()
                        except Exception as e:
                            print(f"Error processing file {img_path}: {e}")

    print('Done!')