import torch
import clip
from PIL import Image
import os
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
##model, preprocess = clip.load("ViT-B/32", device=device)
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)

preprocess = CLIPProcessor.from_pretrained(model_id)

##tokenizing text (classes)
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
##text = clip.tokenize(possible_labels).to(device)

class_descriptions = [
    "drawing",
    "entertaining board game", ##sports game, video game (disjunction of <adjective> game should work fine)
    "graph",
    "emblematic brand logo", 
    "comic",
    "map",
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
    

##for label, prob in zip(possible_labels, probs):
##    print(f"The probability of the image to be {label} is: {prob*100.0:.2f} %")

##Compute accuracy on this test data
labelled_data = []
for image_path in image_file_paths: 
     label = os.path.basename(os.path.dirname(image_path))
     img_data = {
        "img": Image.open(image_path),
        "label": label,
        "label_id": possible_labels.index(label)
     }
     labelled_data.append(img_data)

true_preds = []
for i, img_data in enumerate(labelled_data):
    ##doubt
    if img_data["label_id"] == preds[i]:
        true_preds.append(1)
    else:
        true_preds.append(0)

print(f"Accuracy:", sum(true_preds) / len(true_preds))


##Plotting
x= preds
y= np.zeros(len(preds))
true_labels = [item["label_id"] for item in labelled_data]

tuple_counts = {}

# Iterate through the list of tuples
for tpl in list(zip(preds, true_labels)):
    
    # Use the hashable object as the key in the dictionary and increment the count
    if tpl in tuple_counts:
        tuple_counts[tpl] += 1
    else:
        tuple_counts[tpl] = 1

# Convert the dictionary into a list of tuples
data_list = [(k[0], k[1], v) for k, v in tuple_counts.items()]

# Create a Pandas DataFrame
df = pd.DataFrame(data_list, columns=['prediction', 'true class', 'frequency'])

# Print the DataFrame
#df

sns.scatterplot(data=df, x="prediction", y="true class", size="frequency", legend = False)
p= possible_labels

# Plot the y=x line
plt.plot(p, p, color='red', linestyle='--', label='y=x')


# Show the plot
plt.show()