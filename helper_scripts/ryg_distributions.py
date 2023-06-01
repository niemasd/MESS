#! /usr/bin/env python3
'''
Plot the distribution of red, yellow, and green counts across all pairs of students.

- Red = both students put the same wrong answer
- Yellow = both students got it wrong, but put different wrong answers
- Green = only 1 student got it wrong
'''

# imports
from os.path import isfile
from pandas import DataFrame
from seaborn import histplot, move_legend
from sys import argv, stderr
import matplotlib.pyplot as plt

# throw error
def error(message, prefix="ERROR: ", out_file=stderr):
    print("%s%s" % (prefix, message), file=out_file); exit(1)

# main content
if __name__ == "__main__":
    # parse user args
    if len(argv) != 3:
        error("%s <input_MESS_results_TSV> <output_PDF>" % argv[0], prefix="USAGE: ")
    if not isfile(argv[1].strip()):
        error("Input file not found: %s" % argv[1])
    if isfile(argv[2].strip()):
        error("Output file exists: %s" % argv[2])

    # load responses
    #data = list(); hue = list()
    red, yellow, green = list(), list(), list()
    with open(argv[1]) as tsv:
        for row_num, l in enumerate(tsv):
            if row_num == 0:
                continue
            r, y, g = [int(v.strip()) for v in l.split('\t')[6:9]]
            red.append(r); yellow.append(y); green.append(g)
            #data += [int(v.strip()) for v in l.split('\t')[6:9]]
            #hue += ["Same Wrong Answer", "Different Wrong Answers", "Only 1 Missed"]

    # plot distributions
    fig, ax = plt.subplots(figsize=(10,5))
    max_val = max(max(red),max(yellow),max(green)); bins = list(range(max_val+1))
    histplot(data=red, bins=bins, color='red', label="Same Wrong Answer")
    histplot(data=yellow, bins=bins, color='yellow', label="Different Wrong Answers")
    histplot(data=green, bins=bins, color='green', label="Only 1 Missed")
    plt.xlim(xmin=0, xmax=max_val)
    plt.title("Red, Yellow, and Green Distributions")
    plt.xlabel("Number of Questions")
    plt.ylabel("Number of Pairs of Students")
    plt.legend(bbox_to_anchor=(0.995, 0.995), loc=1, borderaxespad=0., ncol=1)
    plt.savefig(argv[2], format='pdf', bbox_inches='tight')
