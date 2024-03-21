#! /usr/bin/env python3
'''
Plot the distribution of red, yellow, and green counts across all pairs of students.

- Red = both students put the same wrong answer
- Yellow = both students got it wrong, but put different wrong answers
- Green = only 1 student got it wrong
'''

# imports
from os.path import isfile
from matplotlib.backends.backend_pdf import PdfPages
from seaborn import histplot, kdeplot, regplot
from sys import stderr
import argparse
import matplotlib.pyplot as plt

# throw error
def error(message, prefix="ERROR: ", out_file=stderr):
    print("%s%s" % (prefix, message), file=out_file); exit(1)

# main content
if __name__ == "__main__":
    # parse user args
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True, type=str, help="Input MESS Results (TSV)")
    parser.add_argument('-o', '--output', required=True, type=str, help="Output Figures (PDF)")
    parser.add_argument('-t', '--title', required=False, type=str, default="Red, Yellow, and Green Distributions", help="Figure Title")
    args = parser.parse_args()
    args.input = args.input.strip()
    args.output = args.output.strip()

    # check user args
    if not isfile(args.input):
        error("Input file not found: %s" % args.input)
    if isfile(args.output):
        error("Output file exists: %s" % args.output)

    # load responses
    red, yellow, green, total_wrong = list(), list(), list(), list()
    print("Loading data: %s" % args.input)
    with open(args.input) as tsv:
        for row_num, l in enumerate(tsv):
            if row_num == 0:
                continue
            r, y, g = [int(v.strip()) for v in l.split('\t')[6:9]]
            red.append(r); yellow.append(y); green.append(g); total_wrong.append(r+y+g)
    data = [
        (red, 'red', "Same Wrong Answer"),
        (yellow, 'yellow', "Different Wrong Answers"),
        (green, 'green', "Only 1 Missed"),
    ]
    max_val = max(max(red),max(yellow),max(green))
    max_tot = max(total_wrong)
    bins = list(range(max_val+1))

    # plot distributions
    with PdfPages(args.output) as pdf:
        # plot KDE and histogram
        for plot_type_s, plot_type in [("KDE",kdeplot), ("Histogram",histplot)]:
            print("Plotting %s..." % plot_type_s)
            fig, ax = plt.subplots(figsize=(10,5))
            for vals, color, label in data:
                if plot_type is histplot:
                    ylabel = "Number of Pairs"
                    plot_type(data=vals, color=color, label=label, bins=bins)
                else:
                    if color == 'yellow':
                        color = 'goldenrod'
                    ylabel = "Proportion of Pairs"
                    plot_type(data=vals, color=color, label=label, bw_adjust=5)
            plt.xlim(xmin=0, xmax=max_val)
            plt.title(args.title)
            plt.xlabel("Number of Questions")
            plt.ylabel(ylabel)
            plt.legend(bbox_to_anchor=(0.995, 0.995), loc='upper right', borderaxespad=0., ncol=1)
            plt.tight_layout()
            pdf.savefig(plt.gcf()); plt.cla(); plt.clf(); plt.close('all')

        # plot RYG vs. total wrong
        print("Plotting RYG vs. total wrong...")
        fig, ax = plt.subplots(figsize=(10,5))
        for vals, color, label in data:
            kdeplot(x=total_wrong, y=vals, color=color, fill=True, alpha=0.5)#, bw_adjust=1)
        for vals, color, label in data:
            regplot(x=total_wrong, y=vals, color=color, label=label, ci=None, scatter=False)
        plt.xlim(xmin=0, xmax=max_tot)
        plt.ylim(ymin=0, ymax=max_val)
        plt.title(args.title)
        plt.xlabel("Total Number of Questions Either Student Got Wrong (Red+Yellow+Green)")
        plt.ylabel("Number of Questions Red, Yellow, or Green")
        plt.legend(bbox_to_anchor=(0.005, 0.995), loc='upper left', borderaxespad=0., ncol=1)
        plt.tight_layout()
        pdf.savefig(plt.gcf()); plt.cla(); plt.clf(); plt.close('all')
