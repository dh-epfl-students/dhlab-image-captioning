import torch
import clip
from PIL import Image
import os
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import argparse
import scipy.special

parser = argparse.ArgumentParser()
parser.add_argument('--clip', action='store_true', help = "computes new predictions")

parser.add_argument('--create_csv', action='store_true', help= "create a .csv file, is false by default" ) #enter --create_csv to create a .csv file
parser.add_argument('filename', type=str, nargs='?', const=None, help= "enter filename.csv if you wish to create a csv file")
parser.add_argument('--load_csv', type=argparse.FileType('r'), help='Path to the .csv file to be loaded')
args = parser.parse_args()

true_labels = []
preds = []
prefix = "an image of a"
possible_labels = [
    "drawing",
    "game",
    "graph",
    "logo",
    "comic",
    "map",
    "diverse",
    "photo",
    "title",
    "other"]

if args.create_csv and args.filename is None:
    parser.error("If --create_csv is chosen, 'filename' is required.")

if args.load_csv: 
    df = pd.read_csv(args.load_csv)
    true_labels = list(df['True_label'].values)
    preds = list(df['Prediction'].values)
    scores = list(df['Scores'])
elif args.clip: 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ##model, preprocess = clip.load("ViT-B/32", device=device)
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)

    preprocess = CLIPProcessor.from_pretrained(model_id)


    ##tokenizing text (classes)
    
    ##text = clip.tokenize(possible_labels).to(device)

    class_descriptions = [
        "drawing",
        "entertaining board game", ##sports game, video game (disjunction of <adjective> game should work fine)
        "graph",
        "emblematic brand logo", 
        "comic",
        "geographical map", ##trying out disjunction:geographical or meteorological map
        "diverse",
        "photograph",
        "title font letter",
        "other"
    ]

    ##Preprocessing the images
    ##image = preprocess(Image.open(
        ##"data/test/comic/EXP-1956-07-25-a-i0105.jpg")).unsqueeze(0).to(device)
    image_file_paths = []
    for root, dirs, files in os.walk("data"):
        for file in files:
                if file.lower().endswith(".jpg"):
                    image_file_paths.append(os.path.join(root, file))

    images = []
    for image_path in image_file_paths:
        image = Image.open(image_path)
        images.append(image)

    ##contains input ids and attention masks tensors
    class_tokens = preprocess(text=class_descriptions, return_tensors="pt", padding=True).to(device)

    images = preprocess(images=images, return_tensors='pt')['pixel_values'].to(device)

    ##Encode text tokens with sentence embeddings
    with torch.no_grad():
        
        ##text_features = model.encode_text(text)
        class_emb = model.get_text_features(**class_tokens)
        class_emb = class_emb.detach().cpu().numpy()
        ##Normalize class_emb to apply dot product similarity
        class_emb = class_emb / np.linalg.norm(class_emb, axis=0)
        print(f"Shape of class embedding:", class_emb.shape)
        
        ##image_features = model.encode_image(images)
        img_emb = model.get_image_features(images)
        img_emb = img_emb.detach().cpu().numpy()
        print(f"Shape of image embedding:", img_emb.shape)

        ##Computes embeddings for both the image and text inputs simultaneously.
        ##logits_per_image, logits_per_text = model(image, text)

        ##Similarity matrix using dot product 
        preds = []
        scores = np.dot(img_emb, class_emb.T)
        preds.extend(np.argmax(scores, axis=1))

        percentages = scipy.special.softmax(scores, 1)
        

    ##for label, prob in zip(possible_labels, probs):
    ##    print(f"The probability of the image to be {label} is: {prob*100.0:.2f} %")

    ##Compute accuracy on this test data
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
    ##score_tuples = [tuple(row) for row in scores]
    data_dic = {
        "Image_path": img_paths,
        "True_label": true_labels, 
        "Prediction": preds,
        "Scores": np.max(scores, axis=1)
    }
    
    df = pd.DataFrame(data_dic)
    

    if args.create_csv:
        df.to_csv(args.filename)
        

report = sklearn.metrics.classification_report(true_labels, preds)
print(report)

##Confidence level
class_max_prob_sums = np.zeros(len(percentages[0]))
class_data_unit_counts = np.zeros(len(percentages[0]))

for pred, row in zip(preds, percentages):
    percentage = row[pred]
    class_max_prob_sums[pred] += percentage
    class_data_unit_counts[pred] += 1

# Calculate the confidence levels for each class
confidence_levels = np.array(class_max_prob_sums) / np.array(class_data_unit_counts)


print(f"confidence levels:", confidence_levels)

cm = confusion_matrix(
        y_true = true_labels,
        y_pred = preds,
        sample_weight = None,
        normalize= None,
    )
sns.heatmap(cm, vmin= None, vmax=None, cmap='Blues', center=None, robust=None, annot=True, fmt='d', xticklabels= possible_labels , yticklabels= possible_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()