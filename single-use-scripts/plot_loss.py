import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from argparse import ArgumentParser

sns.set_theme()


def read_all_losses_as_dict(train_loss_path, eval_loss_path):
    losses_dict = {}
    for csv_name in os.listdir(eval_loss_path):
        key = csv_name.split('.')[0]
        eval_df = pd.read_csv(os.path.join(eval_loss_path, csv_name))
        train_df = pd.read_csv(os.path.join(train_loss_path, csv_name))
        losses_dict[key] = {'train': train_df, 'eval': eval_df}
    return losses_dict


def plot_losses(losses_pair, figsize=None, title=None, save_path=None):
    if figsize:
        plt.figure(figsize=figsize)
    plt.plot(losses_pair['train'].epoch, losses_pair['train'].loss, losses_pair['eval'].epoch, losses_pair['eval'].eval_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['train', 'eval'])
    if title:
        plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('train_loss_csv_dir', help='path to the directory with training losses .csv files')
    parser.add_argument('eval_loss_csv_dir', help='path to the directory with eval losses .csv files')
    parser.add_argument('-f', '--figs_dir', default=None, help='directory to save loss plots')
    
    args = parser.parse_args()
    losses_dict = read_all_losses_as_dict(args.train_loss_csv_dir, args.eval_loss_csv_dir)
    os.makedirs(args.figs_dir, exist_ok=True)
    for key, loss_pair in losses_dict.items():
        plot_path = os.path.join(args.figs_dir, f'{key}.png')
        plot_losses(loss_pair, title=key, save_path=plot_path)
        plt.clf()