import torch
from clip_repo import clip
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
from natsort import natsorted
from sklearn.metrics import ConfusionMatrixDisplay
import cv2

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


def produce_report(true_labels, preds, max_percentages):
    report = sklearn.metrics.classification_report(
        true_labels, preds, target_names=possible_labels)
    print(report)

    # Confidence levels for each class
    class_max_prob_sums = np.zeros(len(possible_labels))
    class_data_unit_counts = np.zeros(len(possible_labels))
    for pred, percentage in zip(preds, max_percentages):
        class_max_prob_sums[pred] += percentage
        class_data_unit_counts[pred] += 1
    confidence_levels = np.array(
        class_max_prob_sums) / np.array(class_data_unit_counts)

    # Avegerage Confidence level of all classes
    avg_confidence_level = np.average(confidence_levels)

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
    plt.show()

    f1_scores_ = f1_score(true_labels, preds, average=None)
    plt.barh(possible_labels[::-1], f1_scores_[::-1])
    plt.xlabel('F1 Score')
    plt.ylabel('Possible Labels')
    plt.title('F1 score per class prompting CLIP with the class name')

    # Add values on the bars
    for i, score in enumerate(f1_scores_[::-1]):
        plt.text(score, i, f'{score:.2f}', ha='left', va='center')
    plt.show()


def clip_f(class_descriptions, create_csv, filename, type_of_change):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)

    preprocess = CLIPProcessor.from_pretrained(model_id)

    # Preprocessing the images
    image_file_paths = []
    for root, dirs, files in os.walk("data/test"):
        for file in files:
            if file.lower().endswith(".jpg"):
                image_file_paths.append(os.path.join(root, file))

    images = []
    for image_path in image_file_paths:
        image = Image.open(image_path)
        images.append(image)

    # contains input ids and attention masks tensors
    class_tokens = preprocess(
        text=class_descriptions, return_tensors="pt", padding=True).to(device)

    images = preprocess(images=images, return_tensors='pt')[
        'pixel_values'].to(device)

    # Encode text tokens with sentence embeddings
    with torch.no_grad():

        class_emb = model.get_text_features(**class_tokens)
        class_emb = class_emb.detach().cpu().numpy()
        # Normalize class_emb to apply dot product similarity
        class_emb = class_emb / np.linalg.norm(class_emb, axis=0)
        print(f"Shape of class embedding:", class_emb.shape)

        img_emb = model.get_image_features(images)
        img_emb = img_emb.detach().cpu().numpy()
        print(f"Shape of image embedding:", img_emb.shape)

        # Similarity matrix using dot product
        preds = []
        scores = np.dot(img_emb, class_emb.T)
        preds.extend(np.argmax(scores, axis=1))
        max_scores = np.max(scores, axis=1)

        percentages = scipy.special.softmax(scores, 1)
        max_percentages = np.max(percentages, axis=1)

    # Create the labelled data dictionary
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

    if create_csv == True:
        data_dic = {
            "Type_of_change": type_of_change,
            "Image_path": img_paths,
            "True_label": true_labels,
            "Prediction": preds,
            "Scores": max_scores,
            "Confidence": max_percentages,
        }

        df = pd.DataFrame(data_dic)
        df.to_csv('CLIP/spec-paraph-results/'+filename, index=False)


def show_results(folder_path):
    results = []
    accs_per_change = []
    csv_files = [file for file in os.listdir(
        folder_path) if file.endswith('csv')]
    csv_files = sorted(
        csv_files, key=lambda x: int(x.split('_')[1].split('.')[0]))

    def plot_f1_vs_change(csv_files, xlabels):

        for file in csv_files:
            df_change_i = pd.read_csv(os.path.join(folder_path, file))
            f1_scores = sklearn.metrics.f1_score(
                list(df_change_i['True_label']), list(df_change_i['Prediction']), average=None)
            acc = sklearn.metrics.accuracy_score(df_change_i['True_label'], list(
                df_change_i['Prediction']), normalize=True, sample_weight=None)
            results.append(f1_scores)
            accs_per_change.append(acc)
        plt.plot(accs_per_change)
        plt.ylabel('accuracy')
        plt.xticks(ticks=range(len(xlabels)), labels=xlabels)
        plt.xlabel("Class descriptions from simple to more complex")
        for x in range(len(xlabels)):
            plt.axvline(x=x, color='lightsteelblue',
                        linestyle='-', linewidth=1)
        plt.show()

        for i, col in enumerate(range(len(results[0]))):
            if not (i == 6 or i == 9):
                column = [row[col] for row in results]
                plt.plot(column,  label=possible_labels[i])

        for x in range(len(xlabels)):
            plt.axvline(x=x, color='gray', linestyle='--', linewidth=1)
        # Add labels and legend
        plt.xticks(ticks=range(len(xlabels)), labels=xlabels)
        plt.ylabel('F1-score')
        plt.xlabel("Class descriptions from simple to more complex")

        # Display or save the plot
        plt.legend(loc=1)
        plt.show()

    # heat map showing f1 score increase/decrease/stagnation over changes
    def hm_f1_evolution(csv_files, xlabels):
        hm = pd.DataFrame()
        # Results from f1 change - f1 change_0.csv: change this to CLIP/language-results/<language>/change_0_<language>.csv to get this plot for a specific language
        text_is_class_names = pd.read_csv(
            "CLIP/spec-paraph-results/change_0.csv")
        f1_scores_0 = sklearn.metrics.f1_score(list(text_is_class_names['True_label']), list(
            text_is_class_names['Prediction']), average=None)
        for i, file in enumerate(csv_files):
            if i >= 0 and i < 12:
                # Change this path to CLIP/language-results/<language>/ for a specific language
                df_change_i = pd.read_csv(os.path.join(
                    'CLIP/spec-paraph-results/', file))
                f1_scores_i = sklearn.metrics.f1_score(
                    list(df_change_i['True_label']), list(df_change_i['Prediction']), average=None)
                diff = np.array(f1_scores_i - f1_scores_0)
                diff = np.round(diff, 2)
                diff_df = pd.DataFrame(diff)
                # each vector diff is a column of the hm
                hm = pd.concat([hm, diff_df], axis=1)

        hm_plt = sns.heatmap(hm, cmap=sns.diverging_palette(0, 150, as_cmap=True), center=0, annot=True, yticklabels=(
            "drawing", "game", "graph", "logo", "comic", "map", "photo", "title"))
        hm_plt.set_xticklabels(range(1, 13))
        hm_plt.set(
            xlabel="Class descriptions from simple to complex", ylabel="Accuracy")

        plt.show()
        hm_arr = hm.to_numpy()
        average_increase = np.mean(hm_arr, axis=1)

    plot_f1_vs_change(csv_files, range(13))
    hm_f1_evolution(csv_files[1:], range(12))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip', action='store_true',
                        help="Computes new predictions.")

    # enter --create_csv to create a .csv file
    parser.add_argument('--create_csv', action='store_true',
                        help="Create a .csv file, is false by default.")
    parser.add_argument('--load_csv', action='store_true',
                        help='Path to the .csv file to be loaded. Example: spec-paraph-results/change_0.csv or language-results/arabic/change_0_arabic.csv. The report will be printed on the console, the confusion matrix and f1 per class plots will be displayed.')
    parser.add_argument('filename', type=str, nargs='?', const=None,
                        help="Enter filename.csv if you wish to create a csv file.")
    parser.add_argument('type_of_change', type=str, nargs='?', const=None,
                        help="Enter the type of change made to the text descriptions if you wish to create a csv file.")
    parser.add_argument('--open_images', action='store_true',
                        help="Opens a list of images")
    parser.add_argument('--show_results', action='store_true',
                        help="Shows holistic results. Example entry: --show_results x x folder_path")
    parser.add_argument('folder_path', type=str, nargs='?', const=None,
                        help="Enter the folder path containing the csv files to get the holistic results.")
    args = parser.parse_args()
    type_of_change = args.type_of_change
    folder_path = args.folder_path
    print(folder_path)
    true_labels = []
    preds = []
    prefix = "an image of a"

    if args.show_results:
        if folder_path is None:
            print("Specify the folder path if you wish to see the holistic results")
        else:
            show_results(folder_path)

    if args.create_csv and (args.filename is None or args.type_of_change is None):
        parser.error(
            "If --create_csv is chosen, 'filename' and 'type_of_change' are required.")

    if args.load_csv:
        csv_path = os.path.join('CLIP', args.filename)
        df = pd.read_csv(csv_path)
        true_labels = list(df['True_label'].values)
        preds = list(df['Prediction'].values)
        scores = list(df['Scores'].values)
        max_percentages = list(df['Confidence'].values)
        produce_report(true_labels, preds, max_percentages)

    elif args.clip:
        class_descriptions = [
            "non-textual descriptive drawing or illustrative drawing or portrait drawing or caricatural drawing",
            "puzzle game or board game or didactic game",
            "line graph or bar graph",
            "company logo or brand logo or institution logo",
            "adventure comic or humour comic or superhero comic or war comic including text",
            "country map or regional map or weather map",
            "historical photograph or journal photograph or portait photograph",
            "title or alphabet letter"
        ]

        clip_f(class_descriptions, args.create_csv,
               args.filename, args.type_of_change)

    # Understanding missclassification
    if args.open_images:
        image_paths = [
            'data/test/drawing/EXP-1986-07-04-a-i0361.jpg',
            'data/test/drawing/IMP-1949-09-30-a-i0010.jpg',
            'data/test/drawing/IMP-1955-01-31-a-i0053.jpg',
            'data/test/drawing/IMP-1992-02-10-a-i0160.jpg',
            'data/test/drawing/IMP-1911-01-14-a-i0027.jpg',
            'data/test/drawing/IMP-1989-08-28-a-i0011.jpg',
        ]

        for image_path in image_paths:
            if os.path.isfile(image_path):
                # Read the image using OpenCV
                image = cv2.imread(os.path.abspath(image_path))
            else:
                print(f"Unable to read the image at {image_path}")
        else:
            print(f"File not found: {image_path}")
