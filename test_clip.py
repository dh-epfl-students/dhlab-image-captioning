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

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip', action='store_true', help = "Computes new predictions.")

    parser.add_argument('--create_csv', action='store_true', help= "Create a .csv file, is false by default." ) #enter --create_csv to create a .csv file
    parser.add_argument('--add_csv_to_results', action='store_true', help='Appends results of the given .csv to results.csv')
    parser.add_argument('--load_csv', action='store_true', help='Path to the .csv file to be loaded.')
    parser.add_argument('filename', type=str, nargs='?', const=None, help= "Enter filename.csv if you wish to create a csv file or add the content of a csv file to results.csv.")
    parser.add_argument('type_of_change', type=str, nargs='?', const=None, help= "Enter the type of change made to the text descriptions if you wish to create a csv file.")
    
    parser.add_argument('--results', action='store_true', help= "Shows results.")
    
    args = parser.parse_args()
    type_of_change = args.type_of_change

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

    if args.create_csv and (args.filename is None or args.type_of_change is None):
        parser.error("If --create_csv is chosen, 'filename' and 'type_of_change' are required.")

    if (args.add_csv_to_results or args.load_csv) and args.filename is None:
        parser.error("'filename' is required.")

    if args.load_csv or args.add_csv_to_results: 
        df = pd.read_csv(args.filename)
        true_labels = list(df['True_label'].values)
        preds = list(df['Prediction'].values)
        scores = list(df['Scores'].values)
        max_percentages = list(df['Confidence'].values)

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

        possible_changes = [

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
            max_scores = np.max(scores, axis=1)

            percentages = scipy.special.softmax(scores, 1)
            max_percentages = np.max(percentages, axis=1)
            

        ##for label, prob in zip(possible_labels, probs):
        ##    print(f"The probability of the image to be {label} is: {prob*100.0:.2f} %")


        ##Create the labelled data dictionary
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
        
        

        if args.create_csv:
            data_dic = { 
            "Type_of_change": type_of_change,
            "Image_path": img_paths,
            "True_label": true_labels, 
            "Prediction": preds,
            "Scores": max_scores,
            "Confidence": max_percentages,
        }
        
            df = pd.DataFrame(data_dic)
            df.to_csv('experimental-results/'+args.filename, index=False)
            possible_changes.append(type_of_change)

    ##Classification report
    report = sklearn.metrics.classification_report(true_labels, preds)
    print(report)

    ##Confidence levels for each class
    print(f"Confidence level per class:")
    class_max_prob_sums = np.zeros(len(possible_labels))
    class_data_unit_counts = np.zeros(len(possible_labels))
    for pred, percentage in zip(preds, max_percentages):
        class_max_prob_sums[pred] += percentage
        class_data_unit_counts[pred] += 1
    confidence_levels = np.array(class_max_prob_sums) / np.array(class_data_unit_counts)
    for index, confidence_level in enumerate(confidence_levels): 
        print(f"    {possible_labels[index]}: {confidence_level*100.0:.2f} %")

    # Avegerage Confidence level of all classes
    avg_confidence_level = np.average(confidence_levels)
    print(f"The average confidence level is: {avg_confidence_level:.2f}")

    # Accuracy per class
    acc = accuracy_score(true_labels, preds)
    print(f"The average accuracy is:", acc)

    ##Understanding missclassification
    idx_of_diverse = possible_labels.index("diverse")
    ##missclassified data points st true label = diverse but predicted != diverse
    missclassified_diverse = []
    ##missclassified data points st true label != diverse but predicted == diverse
    false_diverse = []
    for true_label, pred, score, conf in zip(true_labels, preds, scores, max_percentages):
        if  true_label == idx_of_diverse and preds != idx_of_diverse:
            missclassified_diverse.append([true_label, pred, score, conf*100])

    missclassified_diverse = pd.DataFrame(missclassified_diverse)
    false_diverse = pd.DataFrame(false_diverse)
    missclassified_diverse.columns = ["True label", "Prediction", "Score", "Confidence in %"]

    ##print(f"Missclassified diverse:")
    ##print(missclassified_diverse)
    ##print(f"Average confidence for missclassified true diverse:", np.average(list(missclassified_diverse['Confidence in %'].values)))

    if args.add_csv_to_results: 
            df_from_csv = pd.read_csv(args.filename)
            #df_from_csv.sort_index(by=['True_label'], ascending=False)
            #df_from_csv.set_index(['Type_of_change', 'True_label'], inplace=True)
            #report = sklearn.metrics.classification_report(true_labels, preds)

            new_exp = {
                "Type_of_change": list(df_from_csv['Type_of_change'])[:10],
                "Class": possible_labels,
                "f1_score": [f'{score:.2f}' for score in f1_score(true_labels, preds, average=None)],
                "Precision": [f'{score:.2f}' for score in sklearn.metrics.precision_score(true_labels, preds, average=None)],
                "Recall": [f'{score:.2f}' for score in sklearn.metrics.recall_score(true_labels, preds, average=None)],
                "Confidence": avg_confidence_level
            }

            new_exp_df = pd.DataFrame(new_exp)
            #Append the new experiment results to results.csv
            new_exp_df.to_csv('results.csv', mode='a', index=False, header=False)
            
    ##Confusion Matrix
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

    
            


        
            

