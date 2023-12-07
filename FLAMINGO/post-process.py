import pandas as pd
from collections import Counter
from test_flam import ask_flamingo, settle_flamingo
import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import argparse
import os
import csv
import shutil

parser = argparse.ArgumentParser(description="Process the captions generated by flamingo to extract the class")
parser.add_argument('--folder_path', type=str, help='Path to the folder with the captions generated. For example: results1')
#parser.add_argument('--num_shots', type=str, help='Integer n from 0 to 3 for n-shots')
parser.add_argument('--add', action='store_true', help='Set this flag to add the results to the results-report')
args = parser.parse_args()
num_shots = args.num_shots
folder_path = args.folder_path
results_file_path = os.path.join(folder_path, "results.csv")
report_file_path = os.path.join(folder_path, "report.csv" )


possible_labels = [
    "title",
    "map",
    "logo",
    "game",
    "photo",
    "graph",
    "drawing",
    "comic",
    "unclas",
]

#filename = "best"
df = pd.read_csv(results_file_path)
true_labels = list(df['Class Name'].values)
image_paths = list(df['File Path'].values) #Should be Image Path
captions = list(df['Predicted Text'].values)
photo_anot = ["photograph", "photo", "view", "close-up"] #screenshot? video? camera?
drawing_anot = ["drawing", "illustration", "fashion", "ornamental", "cartoon"]
game_anot = ["puzzle", "sudoku", "chess", "crosswords", "poker", "card games", "grid", "wordsearch"]
comic_anot = ["comic", "cartoon"]
graph_anot = ["chart", "graph", "plot", "histogram", "diagram", "graphic"]
map_anot = ["map", "blueprint", "plan"]
logo_anot = ["logo", "ad", "advertisement", "publicity", "symbol", "label", "emblem"] #brand?
title_anot = ["title", "letter", "C", "O", "N", "F", "E", "D","R", "Confédéré"]

# Dictionary of words associated to each class
dic = {
    "photo": photo_anot,
    "drawing": drawing_anot,
    "game": game_anot,
    "comic": comic_anot,
    "graph": graph_anot,
    "map": map_anot,
    "logo": logo_anot,
    "title": title_anot,
}

def caption_to_kw(caption):
    pred = []
    #Replace all punctuations by spaces
    caption = caption.replace (".", " ")
    caption = caption.replace (",", " ")
    #print(f"caption: {caption}")
    for word in caption.split():
        for class_name, word_list in dic.items():
            
            if word in word_list:
                #print ("matching words")
                pred.append(class_name)
                #print(f"class_name: {class_name}")
    #print (f"pred: {pred}")
    return pred
    

#This recursive method returns the prediction extracted from the list of keywords extracted from the caption
def filter(index, pred):
    #print(f"pred: {pred}")
    if len(pred) > 1:
        # Words all found in the same class: no confusion in classification
        if len(set(pred)) == 1:
            #print(f"Is this None: {list(set(pred))}")
            filtered_pred = list(set(pred))

        # Words from different classes with no maximum occurrence
        elif len(set(pred)) == len(pred):
            if "drawing" in pred:
                filtered_pred = [item for item in pred if item != "drawing"]
                filtered_pred = filter(index, filtered_pred)
            else: 
                new_prompt = "<image>Between a " + " or a ".join([f'{element}' for element in set(pred)]) + " this image is classified as a"
                print(new_prompt)
                #ask_flamingo(image_paths[index], new_prompt, " vs ".join([f'a "{element}"' for element in set(pred)]) + ".csv", true_labels[index], "2nd-prediction" + filename)
                #print(f"asking flamingo...")
                #print(f"num shots={num_shots}")
                new_caption = settle_flamingo(image_paths[index], new_prompt)
                new_prompt = new_prompt.replace("<image>", '')
                new_caption = new_caption.replace(new_prompt, '')
                print(f"New pred: {new_caption}")
                new_pred = caption_to_kw(new_caption)
                filtered_pred = filter(index, new_pred)

        # Words from different classes with a maximum occurrence = dominant class
        elif len(set(pred)) != len(pred):
            if "drawing" in pred:
                #Removes "drawing"
                filtered_pred = [item for item in pred if item != "drawing"]
                #print(f"filtered_pred: {filtered_pred}")
                filtered_pred = filter(index, filtered_pred) 
            element_counts = Counter(pred)
            filtered_pred = [max(element_counts)]

    elif len(pred) == 0:
        #new_prompt = "<image>Between a comic, a drawing, a game, a graph, a logo, a map, a photograph or a title, this image is classified as a"
        #print(f"new prompt: {new_prompt}")
        #new_caption = settle_flamingo(image_paths[index], new_prompt, num_shots)
        #new_prompt = new_prompt.replace("<image>", '')
        #new_caption = new_caption.replace(new_prompt, '')
        #print(f"new pred: {new_caption}")
        #new_pred = caption_to_kw(new_caption)
        #print(f"inter_pred{new_pred}")
        #filtered_pred = filter(index, new_pred)
        #We try to get a classification twice, if not successful it's unclassified
        #if len(pred) == 0:
            filtered_pred = ["unclas"]

    elif len(pred) == 1: 
        filtered_pred = pred
    print(f"final filtered_pred: {filtered_pred}")
    return filtered_pred

#Prints report, plots, creates post-processed results csv
def produce_report(true_labels, preds):
    
    report = sklearn.metrics.classification_report(
        true_labels, preds, target_names=possible_labels)
    print(report)

    # Accuracy per class
    acc = accuracy_score(true_labels, preds)
    print(f"The average accuracy is:", acc)

    # Confusion Matrix
    cm = confusion_matrix(
        y_true=true_labels,
        y_pred=preds,
        sample_weight=None,
        normalize=None,
    )
    sns.heatmap(cm, vmin=None, vmax=None, cmap='Blues', center=None, robust=None,
                annot=True, fmt='d', xticklabels=possible_labels, yticklabels=possible_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plot_path = os.path.join(folder_path, "ConfusionMatrix.png")
    plt.savefig(plot_path)
    plt.clf()

    #F1-score per class
    f1_scores_ = f1_score(true_labels, preds, average=None)
    plt.barh(possible_labels[::-1], f1_scores_[::-1])
    plt.xlabel('F1 Score')
    plt.ylabel('Possible Labels')
    plt.title('F1 score per class')

    # Add values on the bars
    for i, score in enumerate(f1_scores_[::-1]):
        plt.text(score, i, f'{score:.2f}', ha='left', va='center')
    
    plot_path = os.path.join(folder_path, "F1-scores.png")
    plt.savefig(plot_path)
    

    #Save new results with replacing Predicted Text with Predicted LABEL: in CSV file processed-results.csv
    if not os.path.exists(folder_path):
        print("Error: the results you are trying to process do not exist.")

    csv_file_path = os.path.join(folder_path, "processed-results.csv" )
    shutil.copyfile(results_file_path, csv_file_path)

    df = pd.read_csv(csv_file_path)
    df["Predicted Text"] = preds
    # Write the updated preds back to the CSV file
    df.to_csv(csv_file_path, index=False)


    
    # Open the CSV file in write mode
    with open(report_file_path, 'w') as file:
        file.write(report)

if __name__ == '__main__':

    preds = []
    for caption in captions: 
        pred = caption_to_kw(caption)
        preds.append(pred)
    print(f"Initial prediction: {preds} \n")


    for index, pred in enumerate(preds):
        #print(f"before filtering: {preds[index]}")
        preds[index] = filter(index, pred)
        #print(f"after filtering: {preds[index]}")
    print(f"Processed prediction step 1: {preds} \n")

    #Flatten predictions 
    preds = [item[0] if item else None for item in preds]
    preds = ["unclas" if p is None else p for p in preds]
    print(f"Processed predictions step 2: {preds} \n")
    produce_report(true_labels, preds)
