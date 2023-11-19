import csv
from translate import Translator
import test_clip as tc


def translate_to(language, abr):
    # Initialize a translator
    # Translation can also be made with DeepL Pro when adding: secret = deepL secret key
    translator = Translator(to_lang=abr)

    # Define the input and output file paths
    input_file_path = 'translated-changes-csvs/english_changes2.csv'
    output_file_path = 'translated-changes-csvs/' + language + '_changes.csv'

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

    max_sequence_length = 77

    with open(input_filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        for i in range(1, 13):
            for i, row in enumerate(csv_reader):
                # Process the row and convert it to an array

                # Truncate each cell in the row if it's more than 77 chars
                change_i = [cell[:max_sequence_length] for cell in row[1:]]
                print(len(change_i))
                tc.clip_f(change_i, True, "change_"+str(i)+"_"+language+"csv",
                          type_of_change+str(i))


def clip_results(language):
    type_of_change = language + "_prompts"
    create_results_csv("translated-changes-csvs/" + language +
                       "_changes.csv", language, type_of_change)


if __name__ == "__main__":
    # translate_to("hebrew", "he")
    # clip_results("mandarin")
    tc.show_results("experimental-lang-results/mandarin/")
