import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def draw(output_name, title):
    acc_list = []
    # Iterating through each file and extract the 3 columns we need
    with open(output_name, 'r') as f:
        for line in f:
            json_data = json.loads(line)
            # Extracting the required fields
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)
            score = json_data.get("score", None)
            # Appending to the list
            acc_list.append({
                "Document Depth": document_depth,
                "Context Length": int(context_length.replace('k', '')),
                "score": score
            })
    # Creating a DataFrame
    df = pd.DataFrame(acc_list)

    vmin, vmax = 0.00, 2.00
    from matplotlib.colors import LinearSegmentedColormap
    # Create the pivot table
    pivot_table = pd.pivot_table(df, values='score', index=['Document Depth'], columns=['Context Length'],
                                 aggfunc='mean')

    # Sorting based on 'Context Length'
    df.sort_values('Context Length', inplace=True)

    # Create a custom colormap for the heatmap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with the custom normalization
    plt.figure(figsize=(8, 4))  # Adjust these dimensions as needed
    ax = sns.heatmap(
        pivot_table,
        cmap=cmap,
        vmin=vmin,
        cbar_kws={'label': 'score'}
    )
    # More aesthetics
    plt.title(title)  # Adds a title
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    x_labels = [str(int(label)) + 'k' for label in pivot_table.columns]
    plt.xticks(ticks=np.arange(len(x_labels)) + .5, labels=x_labels, rotation=0)

    # Format the y-ticks to show one decimal place and a percent sign
    y_labels = [f"{label:.1f}%" for label in pivot_table.index]
    ax.set_yticklabels(y_labels)

    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    # Show the plot
    plt.savefig(output_name.replace("jsonl", "png"))
    plt.show()

def infer_subtitle_from_filename(filename):
    # Remove the extension and replace underscores with spaces
    base_name = os.path.splitext(os.path.basename(filename))[0]
    inferred_subtitle = base_name.replace('_', ' ').title()
    return inferred_subtitle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a heatmap from a JSONL file.")
    parser.add_argument("input_file_name", type=str, help="The name of the input JSONL file.")
    parser.add_argument("--subtitle", type=str, default="yentinglin/Llama-3-Taiwan-70B-Instruct-128k", help="The subtitle for the heatmap.")

    args = parser.parse_args()

    main_title = "Needle in a Haystack"
    if args.subtitle is None:
        args.subtitle = infer_subtitle_from_filename(args.input_file_name)

    full_title = f"{main_title}\n{args.subtitle}"

    draw(output_name=args.input_file_name, title=full_title)