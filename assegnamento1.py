#!/usr/bin/env python3
"""First assignment.

Module: basic Python
Assignment #1 (September 30, 2019)


Download a book (not covered by copyright) in plain-text format, e.g.,
from <https://www.gutenberg.org/>

(If you have a hard time picking one, we suggest this English
translation of "The Republic" by Plato:
<http://www.gutenberg.org/cache/epub/1497/pg1497.txt>)


# Goal
Write a Python program that prints the relative frequence of each letter
of the alphabet (without distinguishing between lower and upper case) in
the book.

# Specifications
- the program should have a --help option summarizing the usage
- the program should accept the path to the input file from the command
  line
- the program should print out the total elapsed time
- the program should have an option to display a histogram of the
  frequences
- [optional] the program should have an option to skip the parts of the
  text that do not pertain to the book (e.g., preamble and license)
- [optional] the program should have an option to print out the basic
  book stats (e.g., number of characters, number of words, number of
  lines, etc.)
"""

import argparse
import logging
import time
import string
import os
import re

# A regular expression that matches a single word
RE_WORD = re.compile(r'\b\w+\b')
# A regex that matches the end of Project Gutenberg's preamble
RE_PREAMBLE_END = re.compile(
    r'^\*\*\* START OF THIS PROJECT GUTENBERG.+\*\*\*[\r\n]*$'
)
# A regex that matches the beginning of Project Gutenberg's appendix
RE_APPENDIX_START = re.compile(
    r'^\*\*\* END OF THIS PROJECT GUTENBERG.+\*\*\*[\r\n]*$'
)


def process(file_path, histo=False, stats=False, skip=False):
    """Reads a text file and compile the letter statistics."""
    start_time = time.time()

    logging.info("Reading input file %s...", file_path)
    char_dict = {ch: 0 for ch in string.ascii_lowercase}
    num_chars, num_lines, num_words = 0, 0, 0
    with open(file_path) as input_file:
        line_valid = not skip
        pre_lines = 0
        for line in input_file:
            if line_valid:
                num_lines += 1
                if RE_APPENDIX_START.match(line):
                    logging.info("Beginning of appendix found on line %d",
                                 num_lines + pre_lines)
                    break
                for ch in line.lower():
                    if ch in char_dict:
                        char_dict[ch] += 1
                num_chars += len(line)
                if stats:
                    num_words += len(RE_WORD.findall(line))
            elif RE_PREAMBLE_END.match(line):
                line_valid = True
                pre_lines += 1
                logging.info("End of preamble found on line %d", pre_lines)
            else:
                pre_lines += 1
    logging.info("Done, %d characters found.", num_chars)
    num_letters = sum(char_dict.values())
    if stats:
        print(f"The book has {num_chars} characters, of which"
              f" {num_letters} are letters; it has {num_words} words in"
              f" {num_lines} lines.")

    elapsed_time = time.time() - start_time
    print(f"Done in {elapsed_time:.3f} seconds.")
    for ch, num in char_dict.items():
        print(f"{ch} -> {num / num_letters:.3%}")

    if histo:
        import matplotlib.pyplot as plt
        plt.bar(list(range(len(char_dict))),
                [num / num_letters * 100 for num in char_dict.values()],
                tick_label=list(char_dict.keys()))
        plt.xlabel("Letter")
        plt.ylabel("Relative frequency [%]")
        file_name = os.path.basename(file_path)
        plt.suptitle(f"Relative frequency of letters in {file_name}")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help="Path to the input file")
    parser.add_argument('--debug', '-d', action='store_true',
                        help="Print debug information")
    parser.add_argument('--histo', action='store_true',
                        help="Draw an histogram with matplotlib")
    parser.add_argument('--stats', action='store_true',
                        help="Print statistics about the book")
    parser.add_argument('--skip', action='store_true',
                        help=("Skip metadata (preamble and license) in Project"
                              " Gutenberg files"))
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    process(args.infile, histo=args.histo, stats=args.stats,
            skip=args.skip)
