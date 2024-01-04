import pandas as pd
import random
import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt

# Function to sample 5 images for each label
def sample_images_per_class(df, num_samples=5):
    sampled_data = pd.DataFrame()
    for label, group in df.groupby('Class Name'):
        sampled_group = group.sample(num_samples, random_state=42)
        sampled_data = pd.concat([sampled_data, sampled_group])
    return sampled_data

# Load the classification results CSV files
classification_folder = "FLAMINGO/results/classification"
classification_files = [os.path.join(classification_folder, file) for file in os.listdir(classification_folder)]


# Load the captioning results CSV files
captioning_files = [
    'FLAMINGO/results/captioning/caption1.csv','FLAMINGO/results/captioning/caption2.csv', 'FLAMINGO/results/captioning/caption3.csv'
]

# Create a list to store DataFrames for each type of experiment
classification_dfs = [pd.read_csv(file) for file in classification_files]
captioning_dfs = [pd.read_csv(file) for file in captioning_files]

# Sample 5 images for each label for classification and captioning experiments
common_sampled_df = sample_images_per_class(classification_dfs[0])
samples = common_sampled_df["File Path"]


classification_sampled_dfs = []
for df in classification_dfs:
    sampled_from_df = df[df['File Path'].isin(samples)]
    classification_sampled_dfs.append(sampled_from_df)

captioning_sampled_dfs = []
for df in captioning_dfs:
    sampled_from_df = df[df['File Path'].isin(samples)]
    captioning_sampled_dfs.append(sampled_from_df)


def extract_sort_key(df):
    return int(f"{df['Prompt ID'].iloc[0]}{df['Number of Shots'].iloc[0]}")
class_sorted_samples = sorted(classification_sampled_dfs, key=extract_sort_key)


df = pd.concat(classification_sampled_dfs, axis = 0, ignore_index=True)
df = df.groupby(['File Path']).agg(list)


df_cap = pd.concat(captioning_sampled_dfs, axis = 0, ignore_index=True)
df_cap = df_cap.groupby(['File Path']).agg(list)

#Prepare html for captioning results
html_cap = '<table border="1">'
# Add headers for captioning subcolumns
html_cap += f'<th colspan = "7" style="text-align: center;">Captioning</th>'
html_cap += '<tr>' 
html_cap += f'<th>Index</th><th>File Name</th><th>Image</th><th>Class Name</th><th>Prompt 1: "A complete caption for this image is:"</th><th>Prompt 2: "An image of"</th><th>Prompt 3: "Question: What can you say about the location and time of this image? Answer:"</th>'
html_cap += '</tr>'

#Prepare html for classification results
html_class = '<table border="1">'
# Add headers for classification subcolumns
html_class += f'<th colspan="16" style="text-align: center;">Classification</th></tr>'
html_class += '<tr>' 
html_class += f'<th>Index</th><th>File Name</th><th>Image</th><th>Class Name</th><th colspan = "3">Prompt 1: An image of</th><th colspan = "3">Prompt 2: This image can be classified as</th><th colspan = "3">Prompt 3: Keywords describing this image contain</th><th colspan = "3">Prompt 4: The type of this image is</th>'
html_cap += '</tr>'

html_class += '<tr>'
html_class += f'<th></th><th></th><th></th><th></th>'
for i in range(4):
        html_class += f'<th>0-shot</th><th>1-shot</th><th>2-shot</th>'
html_class += '</tr>'

#Captioning
for index, (i, row) in enumerate(df_cap.iterrows()):
    # Extract information from the row
    img_name = i.split('/')[-1]
    img_path = i
    class_name = row['Class Name']
    try:
        img = plt.imread(img_path[1:])
    except: 
         print('Try img = plt.imread(img_path) instead')
    # Convert image to base64
    img_buffer = BytesIO()
    plt.imsave(img_buffer, img, format='png')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    # HTML row for each image
    html_cap += f'<tr><td>{index}</td><td>{img_name}</td><td><img src="data:image/png;base64,{img_base64}" style="max-width: 300px; max-height: 300px; margin-right: 20px;"></td><td>{class_name[0]}</td>'
    #image --> class name --> prompt --> num_shots
    for pred in row['Predicted Text']:
            html_cap += f'<td>{pred}</td>'
                 
    #pred = classification_values['Predicted Text']
    
    html_cap += '</tr>'
    # Add classification subcolumns
html_cap += '</table>'     

#Classification
for index, (i, row) in enumerate(df.iterrows()):
    # Extract information from the row
    img_name = i.split('/')[-1]
    img_path = i
    class_name = row['Class Name']
    img = plt.imread(img_path)
    
    # Convert image to base64
    img_buffer = BytesIO()
    plt.imsave(img_buffer, img, format='png')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    
    classification_values = row[['Prompt ID', 'Number of Shots', 'Predicted Text']].dropna().to_dict()
    # HTML row for each image
    html_class += f'<tr><td>{index}</td><td>{img_name}</td><td><img src="data:image/png;base64,{img_base64}" style="max-width: 300px; max-height: 300px; margin-right: 20px;"></td><td>{class_name[0]}</td>'
    #image --> class name --> prompt --> num_shots
    for i, pred in enumerate(row['Predicted Text']):
            html_class += f'<td>{pred}</td>'
        
    html_class += '</tr>'
    # Add classification subcolumns
html_class += '</table>'     
            

# Save HTML file
with open('class_summary.html', 'w') as html_file:
    html_file.write(html_class)

with open('cap_summary.html', 'w') as html_file:
    html_file.write(html_cap)
