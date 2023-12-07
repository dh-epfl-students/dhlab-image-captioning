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
#warnings.filterwarnings("ignore")

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
CLASS_PROMPT_LIST = [
    "<image>An image of",
    "<image>This image can be classified as a",
    "<image>Between the classes comic, drawing, game, graph, logo, map, photograph, title, this image is a",
    "<image>Keywords describing this image are",
    "<image>The type of this image is",
]

CAP_PROMPT_LIST = [
    "<image>A complete caption for this image is:"
     "<image>An image of",
]

#prompt = "<image>This image can be classified as a"
#prompt = "<image>The type of this image is"
#prompt = "<image>Between the classes comic, drawing, game, graph, logo, map, photograph, title, this image is a"
#prompt = "<image>Keywords describing this image are"
#misclas_prompt = "<image> Between a logo and a title, this image is classified as a"

def open_demo_imgs(): 
    demo1_comic = Image.open("../data/train/comic/EXP-1925-01-22-a-i0059.jpg")

    demo1_drawing = Image.open("../data/train/drawing/EXP-1900-02-23-a-i0027.jpg")

    demo1_game = Image.open("../data/train/game/EXP-1935-09-13-a-i0054.jpg")

    demo1_graph = Image.open("../data/train/graph/EXP-1903-09-09-a-i0047.jpg")

    demo1_logo = Image.open("../data/train/logo/EXP-1903-12-26-a-i0056.jpg")

    demo1_map = Image.open("../data/train/map/EXP-1939-11-27-a-i0011.jpg")

    demo1_photo = Image.open("../data/train/photo/EXP-1900-12-31-a-i0138.jpg")

    demo1_title = Image.open("../data/train/title/LCE-1900-02-17-a-i0005.jpg")

    img_subset1 = [demo1_comic, demo1_drawing, demo1_game, demo1_graph, demo1_logo, demo1_map, demo1_photo, demo1_title]


    demo2_comic = Image.open("../data/train/comic/EXP-1925-04-23-a-i0091.jpg")

    demo2_drawing = Image.open("../data/train/drawing/EXP-1906-02-21-a-i0051.jpg")

    demo2_game = Image.open("../data/train/game/EXP-1937-12-17-a-i0102.jpg")

    demo2_graph = Image.open("../data/train/graph/EXP-1903-09-12-a-i0043.jpg")

    demo2_logo = Image.open("../data/train/logo/EXP-1904-02-13-a-i0086_2.jpg")

    demo2_map = Image.open("../data/train/map/EXP-1940-05-06-a-i0011.jpg")

    demo2_photo = Image.open("../data/train/photo/EXP-1912-05-29-a-i0089.jpg")

    demo2_title = Image.open("../data/train/title/LCE-1900-03-07-a-i0012.jpg")

    img_subset2 = [demo2_comic, demo2_drawing, demo2_game, demo2_graph, demo2_logo, demo2_map, demo2_photo, demo2_title]


    demo3_comic = Image.open("../data/train/comic/EXP-1956-05-02-a-i0129_3.jpg")

    demo3_drawing = Image.open("../data/train/drawing/EXP-1901-07-17-a-i0091.jpg")

    demo3_game = Image.open("../data/train/game/EXP-1938-06-03-a-i0063.jpg")

    demo3_graph = Image.open("../data/train/graph/EXP-1950-03-24-a-i0151.jpg")

    demo3_logo = Image.open("../data/train/logo/EXP-1915-02-13-a-i0002.jpg")

    demo3_map = Image.open("../data/train/map/EXP-1944-06-20-a-i0012_2.jpg")

    demo3_photo = Image.open("../data/train/photo/EXP-1905-01-12-a-i0048.jpg")

    demo3_title = Image.open("../data/train/title/GDL-1961-10-10-a-i0011.jpg")

    img_subset3 = [demo3_comic, demo3_drawing, demo3_game, demo3_graph, demo3_logo, demo3_map, demo3_photo, demo3_title]

    return img_subset1, img_subset2, img_subset3

def settle_flamingo(img_path, prompt, num_shots): 
    try: 
        demo1_title = Image.open("../data/train/title/LCE-1900-02-17-a-i0005.jpg")
        demo1_game = Image.open("../data/train/game/EXP-1935-09-13-a-i0054.jpg")
        demo1_photo = Image.open("../data/train/photo/EXP-1900-12-31-a-i0138.jpg")
        one_shot_str1 = "<image>Between a logo and a title, this image is classifed as a title.<|endofchunk|><image>Between a graph and a game, this image is classified as a game.<|endofchunk|>"
        #one_shot_str2 = "<image>Between a comic, a drawing, a game, a graph, a logo, a map, a photograph or a title, this image is classified as a photo.<|endofchunk|>"
        adapted_prompt = one_shot_str1 + prompt 
        query_image = Image.open(img_path)
        vision_x = [image_processor(demo1_title).unsqueeze(0), image_processor(demo1_game).unsqueeze(0), image_processor(demo1_photo).unsqueeze(0),  image_processor(query_image).unsqueeze(0)]

        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(device)

        lang_x = tokenizer(
                [adapted_prompt],
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
        last_occurrence_index = decoded_text.rfind('<image>')
        if last_occurrence_index != -1:
            decoded_text = decoded_text[last_occurrence_index + len('<image>'):]
        else:
            decoded_text = decoded_text.strip()

    except Exception as e:
                            print(f"Error processing file {img_path}: {e}")
    return decoded_text

#Ask flamingo to classify. For captioning use num_shots = 0
def ask_flamingo(img_path, prompt, num_shots, s1, s2, s3, prompt_id, caption): 
    
    try: 
        if caption is False:
            one_shot_strs = [
                "<image>An image of a comic.<|endofchunk|><image>An image of a drawing.<|endofchunk|><image>An image of a game.<|endofchunk|><image>An image of a graph.<|endofchunk|><image>An image of a logo.<|endofchunk|><image>An image of a map.<|endofchunk|><image>An image of a photograph.<|endofchunk|><image>An image of a title.<|endofchunk|>",
                "<image>This image can be classifed as a comic.<|endofchunk|><image>This image can be classifed as a drawing.<|endofchunk|><image>This image can be classifed as a game.<|endofchunk|><image>This image can be classifed as a graph.<|endofchunk|><image>This image can be classifed as a logo.<|endofchunk|><image>This image can be classifed as a map.<|endofchunk|><image>This image can be classifed as a photograph.<|endofchunk|><image>This image can be classifed as a title.<|endofchunk|>",
                "<image>Between the classes comic, drawing, game, graph, logo, map, photograph, title, this image is a comic.<|endofchunk|><image>Between the classes comic, drawing, game, graph, logo, map, photograph, title, this image is a drawing.<|endofchunk|><image>Between the classes comic, drawing, game, graph, logo, map, photograph, title, this image is a game.<|endofchunk|><image>Between the classes comic, drawing, game, graph, logo, map, photograph, title, this image is a graph.<|endofchunk|><image>Between the classes comic, drawing, game, graph, logo, map, photograph, title, this image is a logo.<|endofchunk|><image>Between the classes comic, drawing, game, graph, logo, map, photograph, title, this image is a map.<|endofchunk|><image>Between the classes comic, drawing, game, graph, logo, map, photograph, title, this image is a photograph.<|endofchunk|><image>Between the classes comic, drawing, game, graph, logo, map, photograph, title, this image is a title.<|endofchunk|>",
                "<image>Keywords describing this image are comic and cartoon.<|endofchunk|><image>Keywords describing this image are drawing and illustration.<|endofchunk|><image>Keywords describing this image are game and chess.<|endofchunk|><image>Keywords describing this image are graph and chart.<|endofchunk|><image>Keywords describing this image are logo and sign.<|endofchunk|><image>Keywords describing this image are map and plan.<|endofchunk|><image>Keywords describing this image are photograph and view.<|endofchunk|><image>Keywords describing this image are letter and title-font.<|endofchunk|>", #a photo of people
                "<image>The type of this image is a comic.<|endofchunk|><image>The type of this image is a drawing.<|endofchunk|><image>The type of this image is a game.<|endofchunk|><image>The type of this image is a graph.<|endofchunk|><image>The type of this image is a logo.<|endofchunk|><image>The type of this image is a map.<|endofchunk|><image>The type of this image is a photograph.<|endofchunk|><image>The type of this image is a title.<|endofchunk|>",
            ]
            one_shot_str = one_shot_strs[prompt_id -1]
        query_image = Image.open(img_path)
        if num_shots == 0: 
            vision_x = [image_processor(query_image).unsqueeze(0)]
            adapted_prompt = prompt

        elif num_shots == 1: 
            vision_x = [image_processor(image).unsqueeze(0) for image in s1]
            vision_x.append(image_processor(query_image).unsqueeze(0))
            adapted_prompt = one_shot_str + prompt 
        
        elif num_shots == 2:
            vision_x = [image_processor(image).unsqueeze(0) for image in s1]
            vision_x.extend([image_processor(image).unsqueeze(0) for image in s2])
            vision_x.append(image_processor(query_image).unsqueeze(0))
            adapted_prompt = one_shot_str + one_shot_str + prompt

        elif num_shots == 3:
            vision_x = [image_processor(image).unsqueeze(0) for image in s1]
            vision_x.extend([image_processor(image).unsqueeze(0) for image in s2])
            vision_x.extend([image_processor(image).unsqueeze(0) for image in s3])
            vision_x.append(image_processor(query_image).unsqueeze(0))
            adapted_prompt = one_shot_str + one_shot_str + one_shot_str + prompt

        
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(device)

        lang_x = tokenizer(
            [adapted_prompt],
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
        #decoded_text = decoded_text.replace(prompt, '').strip()
        # Remove <|endofchunk|>
        #decoded_text = decoded_text.replace('<|endofchunk|>', '').strip()

        last_occurrence_index = decoded_text.rfind('<image>')
        if last_occurrence_index != -1:
            decoded_text = decoded_text[last_occurrence_index + len('<image>'):]
        else:
            decoded_text = decoded_text.strip()

    except Exception as e:
                            print(f"Error processing file {img_path}: {e}")
    return decoded_text


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate text for images in given classes.")
    parser.add_argument('--data_path', type=str, help='Path to the folder with classes subfolders')
    parser.add_argument('--num_shots', type=int, help='Integer n from 0 to 3 for n-shots')
    parser.add_argument('--prompt_id', type=int, help='Integer n from 1 to 5. The index to prompt mapping is in the file flam_classification_prompts.csv')
    parser.add_argument('--caption', action='store_true', help='Set this flag to true if it is a captioning task. Classification task by default')
    args = parser.parse_args()
    data_path = args.data_path
    num_shots = args.num_shots
    prompt_id = args.prompt_id
    caption = args.caption

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
                            if caption is False: 
                                prompt = CLASS_PROMPT_LIST[prompt_id - 1]
                            else: 
                                 prompt = CAP_PROMPT_LIST[prompt_id - 1]
                            img_subset1, img_subset2, img_subset3 = open_demo_imgs()
                            decoded_text = ask_flamingo(img_path, prompt, num_shots, img_subset1, img_subset2, img_subset3, prompt_id, caption)
                            print(f"Decoded text: {decoded_text}")
                            writer.writerow([img_path, prompt, class_name, decoded_text])
                            file.flush()
                        except Exception as e:
                            print(f"Error processing file {img_path}: {e}")

    print('Done!')