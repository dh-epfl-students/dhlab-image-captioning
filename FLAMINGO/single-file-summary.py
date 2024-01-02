import pandas as pd
import base64
from io import BytesIO
import matplotlib.pyplot as plt

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('FLAMINGO/results/prompt4/1-shot/results.csv')

# Group by Class Name
grouped = df.groupby('Class Name')

# Create a document with 5 random samples for each class
for class_name, group in grouped:
    # Take a random sample of 5 from each class
    group = group.sample(5, random_state=42)

    # Create a new HTML document for each class
    html_output = f'<h2>{class_name} samples</h2>'

    # Plot images and predicted text
    for index, row in group.iterrows():
        try: 
            img_path = row['File Path'][1:]
            img = plt.imread(img_path)
        except:
            print('Try img_path = row[File Path] instead of row[File Path][1:] if the File Path is of the form ./data... instead of ../data...')

        # Convert image to base64
        img_buffer = BytesIO()
        plt.imsave(img_buffer, img, format='png')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        # Embed image in HTML
        html_output += f'<div style="display: flex; align-items: center;">'
        html_output += f'<img src="data:image/png;base64,{img_base64}" alt="{class_name} Image" style="max-width: 300px; max-height: 300px; margin-right: 20px;">'
        html_output += f'<p style="font-size: 16px;">Predicted Text: {row["Predicted Text"]}</p>'
        html_output += f'</div><br>'

    # Save or display the HTML output for each class
    with open(f'{class_name}_samples.html', 'w') as html_file:
        html_file.write(html_output)

