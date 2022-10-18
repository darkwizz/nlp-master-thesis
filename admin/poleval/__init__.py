from argparse import ArgumentParser


def prepare_arg_parser():
    parser = ArgumentParser()
    parser.add_argument('-n', '--nduplicates', type=int, default=0, help='number of possible duplicates of every data item. 0 means only the original item')
    parser.add_argument('-t', '--target_path', required=True, help='destination path of the processed dataset')
    return parser