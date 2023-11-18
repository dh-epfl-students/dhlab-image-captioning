# grab model checkpoint from huggingface hub

from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
from PIL import Image
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,
    # cache_dir="temp"  # Defaults to ~/.cache
    device=device
)

checkpoint_path = hf_hub_download(
    "openflamingo/OpenFlamingo-3B-vitl-mpt1b",
    "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)

model.to(device)

query_image = Image.open("../../data/test/comic/EXP-1956-07-25-a-i0105.jpg")

vision_x = [image_processor(query_image).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)

tokenizer.padding_side = "left"  # For generation padding tokens should be on the left

lang_x = tokenizer(
    ["<image>Classify this image into one of the following classes: comic, drawing, game, graph, logo, map, photo, title. "],
    return_tensors="pt",
)

generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20,
    num_beams=3,
)

print("Generated text: ", tokenizer.decode(generated_text[0]))
