# Exploring Large Vision-Language Pre-trained Models for Historical Images Classification and Captioning

## OVERVIEW

_Context_: The impresso project features a dataset of around 90 digitised historical newspapers containing approximately 3 million images. These images have no labels, and only 10% of them have a caption, two aspects that hinder their retrieval.
Two previous student projects focused on the automatic classification of these images, trying to identify 1. the type of image (e.g. photograph, illustration, drawing, graphic, cartoon, map), and 2. in the case of maps, the country or region of the world represented on that map. Good performances on image type classification were achieved by fine-tuning the VGG-16 pre-trained model (see report).

## GOALS

_Objective_: On the basis of these initial experiments and the annotated dataset compiled on this occasion, the present project will explore recent large-scale language-vision pre-trained models. Specifically, the project will attempt to:

- Evaluate zero-shot image type classification of the available dataset using the CLIP and Flamingo Multi-modal models, and compare with previous performances;
- Explore and evaluate image captioning of the same dataset, including trying to identify countries or regions of the world. This part will require adding caption information on the test part of the dataset. In addition to the fluency and accuracy of the generated captions, a specific aspect that might be taken into account is distinctiveness, i.e. whether the image contains details that differentiate it from similar images.

## **Usage:**

The scripts must be run outside of CLIP/ or FLAMINGO/, i.e from dhlab-image-captioning/ <br />

CLIP:

- To run the test_clip.py, execute the following command in your terminal: <br />

- - python3 CLIP/test_clip.py <br />

- The argument options are: <br />

- - Experiment with different prompts to enter in the class_descriptions variable <br />
- - - --clip <br />
- - When wanting to save the results obtained with particular prompts, the csv will be stored in the directory spec-paraph-results <br />
- - - --clip --create_csv <filename.csv> <type-of-change> <br />
- - When wanting to load a csv file, this command will print the report, and display the confusion matrix and f1 scores <br />
- - - --load_csv <file_path> <br />
- - Opens the list of desired images from the test data, specified inside the image_paths variable
- - - --open_images

- To run the test_languages.py, execute the following command in your terminal, adjusting the main() as needed: <br />
- - python3 CLIP/test_languages.py <br />

FLAMINGO:

- - To run the test_clip.py, execute the following command in your terminal: <br />

## **Student:**

Ines Bouchama

## **Supervisers:**

Emanuela Boros
Maud Ehrmann
