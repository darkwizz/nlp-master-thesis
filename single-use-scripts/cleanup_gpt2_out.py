from tqdm import tqdm
from argparse import ArgumentParser
from utils.data_preprocess import PL_ANSWER_PROMPTS, PROMPT_PLACEHOLDER, QUESTION_PREFIX, ANSWER_PREFIX


def remove_prompt_augmentation(line):
    for prompt in PL_ANSWER_PROMPTS:
        no_placeholder = prompt.rstrip(f'{PROMPT_PLACEHOLDER}.')
        if no_placeholder in line:
            result = line.replace(no_placeholder, '').rstrip('.')
            return result
    return line


def remove_artificial_tags(line):
    result_line = line
    if QUESTION_PREFIX in line:
        result_line = line.split(QUESTION_PREFIX)[1]
    elif ANSWER_PREFIX in line:
        result_line = line.split(ANSWER_PREFIX)[1]
    result_line = result_line.split('</Q')[0]
    result_line = result_line.split('</A')[0]
    return result_line


def main(args):
    source_path = args.source
    destination_path = args.destination

    source_tsv = open(source_path)
    destination_tsv = open(destination_path, 'w')
    for line in tqdm(source_tsv):
        line = line.strip()
        if args.artificial:
            result_line = remove_artificial_tags(line)
        elif args.prompt:
            result_line = remove_prompt_augmentation(line)
        destination_tsv.write(result_line + '\n')
    source_tsv.close()
    destination_tsv.close()
    print(f'Successfully has been written to {destination_path}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--source', required=True, help='path to the source .tsv output file')
    mode_me_group = parser.add_mutually_exclusive_group(required=True)
    mode_me_group.add_argument('-a', '--artificial', action='store_true', help='whether the answers in the source .tsv file are enclosed by artificial <[/]QUESTION> and <[/]ANSWER> tags')
    mode_me_group.add_argument('-p', '--prompt', action='store_true', help='whether the answers in the source .tsv file contain a prompt from the general answers prompt pool')
    parser.add_argument('-d', '--destination', required=True, help='path where the processed .tsv output should be written')
    args = parser.parse_args()
    main(args)