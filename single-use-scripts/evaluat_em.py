import os
import evaluate
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('target_path', help='path to a directory with out.tsv and expected.tsv to calculate ExactMatch')

    args = parser.parse_args()
    if not os.path.exists(args.target_path):
        print('The path does not exist')
        exit(1)
    
    with open(os.path.join(args.target_path, 'expected.tsv')) as expected_file:
        expected = expected_file.readlines()
    
    with open(os.path.join(args.target_path, 'out.tsv')) as out_file:
        out = out_file.readlines()
    
    metric = evaluate.load('exact_match')
    result = metric.compute(predictions=out, references=expected)['exact_match']
    print(f'Exact match: {result:.2f}')