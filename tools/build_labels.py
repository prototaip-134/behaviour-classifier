"""
Script to build the annotations for the rawframes dataset.

Author: Shafquat Tabeeb & Sadat Taseen, 2024.
"""

import os
import argparse
import pandas as pd

SUBSETS = ['train', 'test', 'val']


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, help='path to folder rawframes')
    parser.add_argument('-a', '--annotation_path', type=str, help='path to the annotation excel file')
    parser.add_argument('-ra', '--rawframes', action='store_true', default=False, help='rawframes dataset')
    parser.add_argument('-f', '--file_format', type=str, default='.mp4', help='file format of the videos')
    return parser.parse_args()


def delete_existing_annotations():
    # Delete existing annotation files
    for subset in SUBSETS:
        try:
            os.remove(f'{subset}_annotations.txt')
        except:
            pass


def write_annotations(args):
    root = args.root
    df = pd.read_excel(args.annotation_path)
    for _, video in df.iterrows():
        directory = os.path.join(root, video['video_id'])
        if not args.rawframes: directory += args.file_format
        try:
            labels = video['labels'].split(',')
            # Check if the directory exists
            if not os.path.exists(directory):
                print(f'{directory}, Not found')
                continue

            if args.rawframes:
                frames = len([frame for frame in os.listdir(directory)
                                            if os.path.isfile(os.path.join(directory, frame))])
                line = f"{directory} {frames} {' '.join(labels)}"
            else:
                
                line = f"{directory} {' '.join(labels)}"
                # line = f"{directory} {labels[-1]}"

            print(line)
            with open(f"{video['type']}_annotations.txt", 'a') as fout:
                                fout.write(f"{line}\n")
        except:
            print(f'{directory}, Not found')

if __name__ == '__main__':
    args = load_args()
    delete_existing_annotations()
    write_annotations(args)