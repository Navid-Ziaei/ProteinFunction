import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_go_term_hist(train_terms, show_plot=False, save_fig=True, save_path=''):
    """
    Plot a histogram of the top 100 frequent Gene Ontology (GO) term IDs.

    Parameters:
    - train_terms (pd.DataFrame): A DataFrame containing the GO terms. It should include a column named 'term'.
    - show_plot (bool, optional): If True, the plot will be displayed. Defaults to False.
    - save_fig (bool, optional): If True, the plot will be saved as a PNG file. Defaults to True.
    - save_path (str, optional): The directory path where the plot will be saved. Defaults to an empty string.

    Returns:
    None
    """

    # Extract the top 100 frequent GO terms from the DataFrame
    plot_df = train_terms['term'].value_counts().iloc[:100]

    # Create a bar plot for the top 100 GO terms
    figure, axis = plt.subplots(1, 1, figsize=(12, 6))
    bp = sns.barplot(ax=axis, x=np.array(plot_df.index), y=plot_df.values)

    # Set the x-axis labels, rotation, and font size
    bp.set_xticklabels(bp.get_xticklabels(), rotation=90, size=6)

    # Set the title and axis labels
    axis.set_title('Top 100 frequent GO term IDs')
    bp.set_xlabel("GO term IDs", fontsize=12)
    bp.set_ylabel("Count", fontsize=12)

    # Display the plot if show_plot is True
    if show_plot:
        plt.show()

    # Save the plot as a PNG file if save_fig is True
    if save_fig:
        figure.savefig(save_path + "Top_100_frequent_GO_term_IDs.png")


def plot_aspects_pie_plot(train_terms, show_plot=False, save_fig=True, save_path=''):
    # Create a bar plot for the top 100 GO terms
    figure, axis = plt.subplots(1, 1, figsize=(6, 6))
    pie_df = train_terms['aspect'].value_counts()
    palette_color = sns.color_palette('bright')
    plt.pie(pie_df.values, labels=np.array(pie_df.index), colors=palette_color, autopct='%.0f%%')
    # Display the plot if show_plot is True
    if show_plot:
        plt.show()

    # Save the plot as a PNG file if save_fig is True
    if save_fig:
        figure.savefig(save_path + "Top_100_frequent_GO_term_IDs.png")


def plot_history(history, show_plot=False, save_fig=True, save_path=''):
    figure, axis = plt.subplots(1, 1, figsize=(6, 6))
    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss']].plot(title="Cross-entropy", ax=axis)
    history_df.loc[:, ['binary_accuracy']].plot(title="Accuracy", ax=axis)
    # Display the plot if show_plot is True
    if show_plot:
        plt.show()

    # Save the plot as a PNG file if save_fig is True
    if save_fig:
        figure.savefig(save_path + "Training_curves.png")