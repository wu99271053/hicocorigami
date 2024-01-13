import os
from tqdm import tqdm
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt


def get_colormap(itype):
    colormaps = {
        "Inward": "cool",
        "Outward": "winter",
        "Tandem+": "autumn",
        "Tandem-": "summer",
        "Combined": "Wistia"
    }

    return colormaps.get(itype, "cool") 

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description='C.Origami testing Module.')

    parser.add_argument('--val_chr', dest='val_chr', default=1,
                            type=int,
                            help='Random seed for training')

    parser.add_argument('--itype', default='Tandem+',
                            help='Path to the model checkpoint')

    # Data directories
    parser.add_argument('--data_root', default='Desktop/results',
                            help='Root path of training data')
    
    parser.add_argument('--window', dest='window', default=64,
                            type=int,
                            help='Random seed for training')
    


    args = parser.parse_args()

    outputlist_df = pd.read_csv(f"{args.data_root}/{args.itype}/{args.window}/{args.val_chr}/result/computed_outputs.csv", header=None)
    targetlist_df = pd.read_csv(f"{args.data_root}/{args.itype}/{args.window}/{args.val_chr}/result/targets.csv", header=None)
    untrain_outputlist_df = pd.read_csv(f"{args.data_root}/{args.itype}/{args.window}/{args.val_chr}/result/untrain_outputs.csv", header=None)

    # Converting DataFrames to NumPy arrays and reshaping
    outputlist = outputlist_df.values.reshape(-1, args.window, args.window)
    targetlist = targetlist_df.values.reshape(-1, args.window, args.window)
    untrain_outputlist = untrain_outputlist_df.values.reshape(-1, args.window, args.window)
    colormap = get_colormap(args.itype)
    save_dir=f"{args.data_root}/{args.itype}/{args.window}/{args.val_chr}/figs/"
    os.mkdir(save_dir)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # Adjusted size for faster processing
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in tqdm(range(len(outputlist))):
        for ax in axes:
            ax.clear()
        prediction = outputlist[i]
        truth = targetlist[i]
        untrained = untrain_outputlist[i]
        im1 = axes[0].imshow(prediction, cmap=colormap, interpolation='bilinear')
        axes[0].set_title('Predicted')
        im2 = axes[1].imshow(truth, cmap=colormap, interpolation='bilinear')
        axes[1].set_title('Truth')
        im3 = axes[2].imshow(untrained, cmap=colormap, interpolation='bilinear')
        axes[2].set_title('Untrained')

        plt.savefig(os.path.join(save_dir, f'heatmap_{i}.jpg'),format='jpg')
        
    plt.close(fig)  # Close the figure to free memory

    spearman_scores_pred_truth = []
    pearson_scores_pred_truth = []
    spearman_scores_untrain_truth = []
    pearson_scores_untrain_truth = []

    for i in range(len(outputlist)):
        prediction = outputlist[i].flatten()
        truth = targetlist[i].flatten()
        untrained = untrain_outputlist[i].flatten()

        # Compute Spearman and Pearson correlations
        spearman_pred_truth, _ = stats.spearmanr(prediction, truth)
        pearson_pred_truth, _ = stats.pearsonr(prediction, truth)
        spearman_untrain_truth, _ = stats.spearmanr(untrained, truth)
        pearson_untrain_truth, _ = stats.pearsonr(untrained, truth)

        # Store the scores
        spearman_scores_pred_truth.append(spearman_pred_truth)
        pearson_scores_pred_truth.append(pearson_pred_truth)
        spearman_scores_untrain_truth.append(spearman_untrain_truth)
        pearson_scores_untrain_truth.append(pearson_untrain_truth)

    # Create a box plot for the correlation scores
    data_to_plot = [spearman_scores_pred_truth, spearman_scores_untrain_truth,
                    pearson_scores_pred_truth, pearson_scores_untrain_truth]

    # Colors for Prediction and Untrained
    colors = ['blue', 'green', 'blue', 'green']

    plt.figure(figsize=(10, 6))

    # Creating box plots
    bp = plt.boxplot(data_to_plot, patch_artist=True, positions=[1, 2, 4, 5], widths=0.6)

    # Apply colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Adding legend
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=colors[i], label=['Prediction', 'Untrained'][i % 2]) for i in range(2)]
    plt.legend(handles=legend_patches, loc='upper right')

    plt.xticks([1.5, 4.5], ['Spearman Correlation', 'Pearson Correlation'])
    plt.ylabel('Correlation Score')
    plt.title("Combined")
    plt.savefig(os.path.join(save_dir, 'correlation_scores_boxplot.png'))
    plt.close()