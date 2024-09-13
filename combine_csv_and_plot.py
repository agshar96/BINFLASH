import pandas as pd
import matplotlib.pyplot as plt
import glob
from matplotlib.ticker import MaxNLocator
import os

def plot_combined(df, path):
    # Plot the DataFrame
    x_values = df.iloc[:, 0]

    # Plot each remaining column as a separate line
    color = ['#D55E00', '#0072B2', '#009E73', '#CC79A7']
    plt.figure(figsize=(9, 6))
    for column in df.columns[1:]:
        plt.plot(x_values, df[column], label=column, color=color.pop(0))
    # Add labels and title
    plt.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    # plt.tight_layout()
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(path + '/combined_plot.png')
    plt.close()

def process_csv(path, RCM=False):
    # Get all CSV files in the directory
    print("Processing: ", path)
    csvpath = str(path) + '/*.csv'
    csv_files = glob.glob(csvpath)  # Update this to your CSV file path or use a list of specific files

    # Initialize a DataFrame by loading the first CSV file
    df_output = pd.read_csv(csv_files[0])
    for file in csv_files[1:]:
        if 'combined' in file:
            continue
        df = pd.read_csv(file)
        df_output['Base Flash Attention'] += df['Base Flash Attention']
        df_output['Naive Attention Masking'] += df['Naive Attention Masking']
        df_output['Binary Block Masking'] += df['Binary Block Masking']
        if RCM:
            df_output['Binary Block Masking with RCM'] += df['Binary Block Masking with RCM']

    plot_combined(df_output, path)
    start_name = csv_files[0].split('/')[-1].split('-')[0]
    output_name = start_name + '-combined.csv'
    output_name = os.path.join(path, output_name)
    df_output.to_csv(output_name, index=False)
    
base_path = 'benchresult_applications'

for dir in os.listdir(base_path):
    if 'kernels' in str(dir) or 'Sparse' in str(dir):
        continue
    if dir == 'LongFormer':
        path = os.path.join(base_path, dir)
        for subdir in os.listdir(path):
            subpath = os.path.join(path, subdir)
            process_csv(subpath)
    elif 'RCM' in str(dir):
        path = os.path.join(base_path, dir)
        if os.path.isdir(path):
            process_csv(path, RCM=True)
    else:
        path = os.path.join(base_path, dir)
        if os.path.isdir(path):
            process_csv(path)
        
            

                
        