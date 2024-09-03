import os
import argparse
from utils.preprocess import ImageCleaner

def main():
    parser = argparse.ArgumentParser(description='Process images from input directory and save cleaned images to output directory while mimicking the directory structure.')
    parser.add_argument('input_dir', type=str, help='Path to the input directory containing images.')
    parser.add_argument('output_dir', type=str, help='Path to the output directory where cleaned images will be saved.')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='Number of worker processes to use for parallel processing.')

    args = parser.parse_args()

    cleaner = ImageCleaner(args.workers)
    cleaner.process_directory(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()
