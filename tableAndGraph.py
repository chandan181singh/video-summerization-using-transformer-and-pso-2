import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data for the tables
summe_data = {
    "Method": [
        "SUM-GAN dpp [11]",
        "vsLSTM [13]",
        "dppLSTM [13]",
        "SUM-GAN sup [11]",
        "DR-DSN [14]",
        "Two-stream Transformer [21]",
        "Multimodal Transformer [22]",
        "A-AVS",
        "Proposed transformer-pso-based"
    ],
    "Learning": [
        "Unsupervised",
        "Supervised",
        "Supervised",
        "Supervised",
        "Supervised",
        "Supervised",
        "Supervised",
        "Supervised",
        "Supervised"
    ],
    "F-Score": [39.1, 37.6, 38.6, 41.7, 42.1, 39.4, 41.5, 48.1, 47.3]
}


tvsum_data = {
    "Method": [
        "TVSum [12]",
        "SUM-GAN dpp [11]",
        "vsLSTM [13]",
        "dppLSTM [13]",
        "SUM-GAN sup [11]",
        "DR-DSN [14]",
        "Two-stream Transformer [21]",
        "Multimodal Transformer [22]",
        "A-AVS",
        "Proposed transformer-pso-based"
    ],
    "Learning": [
        "Unsupervised",
        "Unsupervised",
        "Supervised",
        "Supervised",
        "Supervised",
        "Supervised",
        "Supervised",
        "Supervised",
        "Supervised",
        "Supervised"
    ],
    "F-Score": [51.3, 51.7, 54.2, 54.7, 56.3, 58.1, 56.2, 57.9, 52.4, 52.6]
}

# Creating tables
summe_df = pd.DataFrame(summe_data)
tvsum_df = pd.DataFrame(tvsum_data)

# Updated Graph Data
x = np.array(["TVSum", "SumMe"])
tps = [52.4, 48.1] 

# Plotting the graph
plt.figure(figsize=(8, 6))
width = 0.35  # Increased width since we only have 2 bars now

x_indexes = np.arange(len(x))
plt.bar(x_indexes - width/2, tps, width=width, label="transformer-pso", color='#2ecc71')

plt.xlabel("Datasets", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.title("Performance Comparison on different datasets", fontsize=14)
plt.xticks(ticks=x_indexes, labels=x)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of each bar
for i in range(len(x)):
    plt.text(i - width/2, tps[i] + 0.5, f'{tps[i]}', 
             ha='center', va='bottom')

plt.ylim(0, max(max(tps), 0) + 5)  # Add some padding above bars
plt.tight_layout()

# Save graph and display
plt.savefig("proposed_methods_comparison.png")
plt.show()

# Create tables as images using matplotlib

# Function to create table image
def create_table_image(data, title, filename):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')
    
    # Convert values to strings and make proposed work bold
    cell_text = data.values.copy()
    for i in range(len(data)):
        for j in range(len(data.columns)):
            if 'Proposed' in str(cell_text[i][0]):  # Check if it's a proposed method row
                # Add explicit space using \: in LaTeX
                text = str(cell_text[i][j]).replace("Proposed ", "Proposed\\:")
                cell_text[i][j] = f'$\\bf{{{text}}}$'
    
    # Create table
    table = ax.table(cellText=cell_text,
                    colLabels=data.columns,
                    cellLoc='center',
                    loc='center')
    
    # Adjust table style
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Make column headers bold
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_text_props(weight='bold')
    
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

# Create and save SumMe table
create_table_image(summe_df, '', 'summe_table.png')

# Create and save TVSum table
create_table_image(tvsum_df, '', 'tvsum_table.png')
