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
    
    parser.add_argument('--gaussian',action='store_true',
                        help='processed data and saved checkpoint')


    args = parser.parse_args()


    if args.gaussian:
        data_dir=f'{args.itype}/gaussian/{args.window}/checkpoint_{args.val_chr}/inference_result'
    else:
        data_dir=f'{args.itype}/notransform/{args.window}/checkpoint_{args.val_chr}/inference_result'

    timestep0 = pd.read_csv(f"{data_dir}/prediction_0.csv", header=None)
    timestep4 = pd.read_csv(f"{data_dir}/prediction_4.csv", header=None)
    timestep8 = pd.read_csv(f"{data_dir}/prediction_8.csv", header=None)
    timestep15 = pd.read_csv(f"{data_dir}/prediction_15.csv", header=None)
    timestep30 = pd.read_csv(f"{data_dir}/prediction_30.csv", header=None)
    timestep60 = pd.read_csv(f"{data_dir}/prediction_60.csv", header=None)

    time0list = timestep0.values.reshape(-1, args.window, args.window)
    time4list = timestep4.values.reshape(-1, args.window, args.window)
    time8list = timestep8.values.reshape(-1, args.window, args.window)
    time15list = timestep15.values.reshape(-1, args.window, args.window)
    time30list = timestep30.values.reshape(-1, args.window, args.window)
    time60list = timestep60.values.reshape(-1, args.window, args.window)


    colormap = get_colormap(args.itype)
    if args.gaussian:
        save_dir=f'{args.itype}/gaussian/{args.window}/checkpoint_{args.val_chr}/inference_figs/'
    else:
        save_dir=f'{args.itype}/notransform/{args.window}/checkpoint_{args.val_chr}/inference_figs/'

    os.mkdir(save_dir)


    fig, axes = plt.subplots(1, 6, figsize=(12, 4))  # Adjusted size for faster processing
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in tqdm(range(len(time0list))):
        for ax in axes:
            ax.clear()
        time0 = time0list[i]
        time4 = time4list[i]
        time8 = time8list[i]
        time15 = time15list[i]
        time30 = time30list[i]
        time60 = time60list[i]

        im1 = axes[0].imshow(time0, cmap=colormap, interpolation='bilinear')
        axes[0].set_title('Time0')
        im2 = axes[1].imshow(time4, cmap=colormap, interpolation='bilinear')
        axes[1].set_title('Time4')
        im3 = axes[2].imshow(time8, cmap=colormap, interpolation='bilinear')
        axes[2].set_title('Time8')
        im4 = axes[3].imshow(time15, cmap=colormap, interpolation='bilinear')
        axes[3].set_title('Time15')
        im5 = axes[4].imshow(time30, cmap=colormap, interpolation='bilinear')
        axes[4].set_title('Time30')
        im6 = axes[5].imshow(time60, cmap=colormap, interpolation='bilinear')
        axes[5].set_title('Time60')

        plt.savefig(os.path.join(save_dir, f'heatmap_{i}.jpg'),format='jpg')
        
    plt.close(fig)  # Close the figure to free memory







