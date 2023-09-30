import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open(
    "data/test/comic/EXP-1956-07-25-a-i0105.jpg")).unsqueeze(0).to(device)

possible_labels = [
    "a drawing",
    "a game",
    "a graph",
    "a logo",
    "a comic",
    "a map",
    "diverse",
    "a photo",
    "a title",
    "other"]

text = clip.tokenize(possible_labels).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

for label, prob in zip(possible_labels, probs[0]):
    print(f"The probability of the image to be {label} is: {prob*100.0:.2f} %")
