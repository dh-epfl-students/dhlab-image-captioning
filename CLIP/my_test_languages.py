import csv
from translate import Translator
import test_clip as tc

# Translate all prompts to the desired language


def translate_to(language, abr):
    # Initialize a translator
    # Translation can also be made with DeepL Pro when adding: secret = deepL secret key
    translator = Translator(to_lang=abr)

    # Define the input and output file paths
    input_file_path = 'CLIP/translations/english_changes.csv'
    output_file_path = 'CLIP/translations/' + language + '_changes.csv'

    # Read the input CSV file and create an output CSV file
    with open(input_file_path, 'r', newline='') as input_file, open(output_file_path, 'w', newline='') as output_file:
        csv_reader = csv.reader(input_file)
        csv_writer = csv.writer(output_file)

        for row in csv_reader:

            # Maximum allowed sequence length by the model
            max_sequence_length = 77

            # Translate each cell in the row while truncating the input text
            translated_row = [translator.translate(
                cell[:max_sequence_length]) for cell in row]

            csv_writer.writerow(translated_row)


def create_results_csv(input_filename, language, type_of_change):
    # Adapt the maximum sequence length depending on needs and language
    max_sequence_length = 77

    with open(input_filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        for i in range(1, 13):
            # Process each row and convert it to an array
            for i, row in enumerate(csv_reader):
                # Truncate each cell in the row if it's more than 77 chars
                change_i = [cell[:max_sequence_length] for cell in row[1:]]
                tc.clip_f(change_i, True, "change_"+str(i)+"_"+language+"csv",
                          type_of_change+str(i))


# Create a csv file with the results of a specific language for all prompts, after having translated all prompts


def clip_results(language):
    type_of_change = language + "_prompts"
    create_results_csv("CLIP/translations/" + language +
                       "_changes.csv", language, type_of_change)


if __name__ == "__main__":
    translate_to("arabic", "ar")
    clip_results("arabic")

    try:
        tc.show_results("CLIP/language-results/arabic/")
    except:
        print('Please follow the instructions inside the method show_results() in test_clip.py to show language results')
