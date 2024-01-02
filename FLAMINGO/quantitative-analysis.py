import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon
from scipy.optimize import minimize

# Create a new folder for plots if it doesn't exist
plots_folder = 'FLAMINGO/freq-plots'
os.makedirs(plots_folder, exist_ok=True)


# Iterate over each CSV file in the class_res folder
class_res_folder = 'FLAMINGO/results/classification'
for filename in os.listdir(class_res_folder):
    if filename.endswith('.csv'):
        # Read the CSV file into a pandas DataFrame
        filepath = os.path.join(class_res_folder, filename)
        df = pd.read_csv(filepath)

        # Extract relevant columns
        prompt_id_col = 'Prompt ID'
        num_shots_col = 'Number of Shots'
        predicted_text_col = 'Predicted Text'

        # Perform quantitative analysis - frequency analysis
        freq_df = df[predicted_text_col].value_counts().reset_index()
        freq_df.columns = ['Predicted Text', 'Frequency']
        frequencies = freq_df['Frequency'].values

        # Plot and save the frequency vs predicted text plot
        prompt_id = df[prompt_id_col].iloc[0]
        num_shots = df[num_shots_col].iloc[0]

        plt.bar(freq_df['Predicted Text'], freq_df['Frequency'])
        plt.xlabel('')
        plt.ylabel('Frequency')
        plt.title(f'Frequency of predicted texts with {num_shots}-shot prompt {prompt_id}')

        # Apply log scale to the y-axis
        plt.yscale('log')

        plt.xticks([])
        plt.tight_layout()

        plot_filename = f'{num_shots}-shot_prompt_{prompt_id}_freq_plot.png'
        plot_filepath = os.path.join(plots_folder, plot_filename)
        plt.savefig(plot_filepath)
        plt.close()

        print(f'Plot saved: {plot_filepath}')

