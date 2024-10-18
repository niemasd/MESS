#! /usr/bin/env python3
'''
Plot the distribution of student responses for each question from a spreadsheet of student responses in the MESS TSV format.
'''

# imports
from collections import Counter
from csv import reader
from matplotlib.backends.backend_pdf import PdfPages
from os.path import isfile
from seaborn import displot
from sys import argv, stderr
from warnings import filterwarnings
import matplotlib.pyplot as plt
filterwarnings("error")

# throw error
def error(message, prefix="ERROR: ", out_file=stderr):
    print("%s%s" % (prefix, message), file=out_file); exit(1)

# load responses from MESS TSV
def load_mess_responses(mess_tsv_fn):
    print("Loading data from: %s" % mess_tsv_fn)
    questions = None # list of question labels
    responses = dict() # responses[email][i] = response from student `email` for question `questions[i]`
    with open(mess_tsv_fn) as mess_tsv:
        for row_num, row in enumerate(reader(mess_tsv, delimiter='\t', quotechar='"')):
            email = row[0].strip()
            question_cols = [v.strip() for v in row[2:]]
            if row_num == 0:
                questions = question_cols
            else:
                if email in responses:
                    error("Duplicate student: %s" % email)
                responses[email] = question_cols
    if questions is None:
        error("Input file is empty: %s" % mess_tsv_fn)
    if len(questions) == 0:
        error("No question columns in input file: %s" % mess_tsv_fn)
    if len(responses) == 0:
        error("No student rows in input file: %s" % mess_tsv_fn)
    return questions, responses

# plot response distributions
def plot_response_dists(questions, responses, pdf_fn, xlabel="Response", ylabel="Count", yscale="linear", aspect=2, xtick_rotation=90):
    with PdfPages(pdf_fn) as pdf:
        for i, q in enumerate(questions):
            # parse current question's responses
            print("Question %d of %d..." % (i+1, len(questions)), end='\r')
            curr_responses = [responses[email][i] for email in responses if len(responses[email][i]) != 0]
            curr_counts = Counter(curr_responses)
            curr_order = sorted(set(curr_responses), key=lambda x: curr_counts[x], reverse=True)
            curr_responses = [v for v in curr_order for _ in range(curr_counts[v])] # sort in descending order of count
            if len(curr_responses) == 0:
                continue # every response is empty (e.g. Parson problems on EdStem, which don't export to the output file)

            # create current plot
            fg = displot(data=curr_responses, aspect=aspect)
            plt.title(q)
            plt.xlabel(xlabel)
            ymin = 0
            if yscale != 'linear':
                ylabel += (' (%s-scale)' % yscale)
                if yscale == 'log':
                    ymin = 1
            plt.ylim(ymin=ymin)
            plt.ylabel(ylabel)
            plt.yscale(yscale)
            plt.xticks(rotation=xtick_rotation)

            # add counts above each bar
            spots = zip(fg.ax.patches, curr_order)
            for spot in spots:
                fg.ax.text(spot[0].get_x()+spot[0].get_width()/4, spot[0].get_height(), str(curr_counts[spot[1]]))

            # finalize and save current plot
            try:
                plt.tight_layout()
            except UserWarning:
                plt.gcf().subplots_adjust(bottom=0.5, top=0.92)
            pdf.savefig(plt.gcf())
            plt.cla(); plt.clf(); plt.close('all')
    print("Response distribution plots written to: %s" % pdf_fn)

# main content
if __name__ == "__main__":
    # parse user args
    if len(argv) != 3:
        error("%s <input_MESS_responses_TSV> <output_PDF>" % argv[0], prefix="USAGE: ")
    if not isfile(argv[1].strip()):
        error("Input file not found: %s" % argv[1])
    if isfile(argv[2].strip()):
        error("Output file exists: %s" % argv[2])

    # load responses and plot distributions
    questions, responses = load_mess_responses(argv[1])
    plot_response_dists(questions, responses, argv[2])
