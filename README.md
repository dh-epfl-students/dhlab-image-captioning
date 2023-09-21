# Exploring Large Vision-Language Pre-trained Models for Historical Images Classification and Captioning

## OVERVIEW

*Context*: The impresso project features a dataset of around 90 digitised historical newspapers containing approximately 3 million images. These images have no labels, and only 10% of them have a caption, two aspects that hinder their retrieval.
Two previous student projects focused on the automatic classification of these images, trying to identify 1. the type of image (e.g. photograph, illustration, drawing, graphic, cartoon, map), and 2. in the case of maps, the country or region of the world represented on that map. Good performances on image type classification were achieved by fine-tuning the VGG-16 pre-trained model (see report).

## GOALS

*Objective*: On the basis of these initial experiments and the annotated dataset compiled on this occasion, the present project will explore recent large-scale language-vision pre-trained models. Specifically, the project will attempt to: 
- Evaluate zero-shot image type classification of the available dataset using the CLIP and LLaMA-Adapter-V2 Multi-modal models, and compare with previous performances;
- Explore and evaluate image captioning of the same dataset, including trying to identify countries or regions of the world. This part will require adding caption information on the test part of the dataset. In addition to the fluency and accuracy of the generated captions, a specific aspect that might be taken into account is distinctiveness, i.e. whether the image contains details that differentiate it from similar images.
