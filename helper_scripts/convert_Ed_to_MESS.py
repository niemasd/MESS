#! /usr/bin/env python3
'''
Convert student responses exported by the Ed platform ("Download Lesson Quiz Responses") into the MESS TSV format.

Columns assumed in Ed TSVs:
Name, Email, Total Score, Q1 score, Q2 score, ..., Qn score, Q1, Q1 submitted, Q2, Q2 submitted, ..., Qn, Qn submitted
'''

# imports
from csv import reader, writer
from os.path import isfile
from sys import argv, stderr

# throw error
def error(message, prefix="ERROR: ", out_file=stderr):
    print("%s%s" % (prefix, message), file=out_file); exit(1)

# load responses from Ed CSV(s)
def load_ed_responses(ed_csv_fns):
    # prepare for loading data
    questions = dict() # questions[fn] = list of question labels for `fn`
    responses = dict() # responses[fn][email] = list of responses from student `email` for `fn`
    correct   = dict() # correct[fn][email] = list of correct question indices from student `email` for `fn`

    # load data and return
    for csv_fn in ed_csv_fns:
        n = None
        with open(csv_fn) as csv_f:
            for row_num, row in enumerate(reader(csv_f)):
                # header row, so initialize this CSV's entry
                if row_num == 0:
                    n = int((len(row)-3)/3)
                    questions[csv_fn] = [(i,row[i].strip()) for i in range(3+n, len(row), 2)] # list of (q_ind, q_name) tuples
                    responses[csv_fn] = dict(); correct[csv_fn] = dict()
                    for q in questions[csv_fn]:
                        if ',' in q:
                            error("Question labels cannot have commas: %s" % q)
                    continue

                # parse student row
                email = row[1].strip()
                if email in responses[csv_fn]:
                    error("Duplicate email found: %s in file %s" % (email, csv_fn))
                responses[csv_fn][email] = [row[qi].strip() for qi,q in questions[csv_fn]]
                correct[csv_fn][email] = [i for i,v in enumerate(row[3:3+n]) if int(v) == 1] # 1 is assumed to be max score
    return questions, responses, correct

# main content
if __name__ == "__main__":
    # parse user args
    if len(argv) < 3:
        error("%s <output_MESS_TSV> <input_Ed_CSV> [input_Ed_CSV_2] ..." % argv[0], prefix="USAGE: ")
    if isfile(argv[1].strip()):
        error("Output file exists: %s" % argv[1])
    unique_fn = set()
    for curr_fn in argv[2:]:
        fn = curr_fn.strip()
        if ',' in fn:
            error("Input filenames cannot have commas: %s" % fn)
        if fn in unique_fn:
            error("Duplicate input file: %s" % fn)
        if not isfile(fn):
            error("Input file not found: %s" % fn)
        unique_fn.add(fn)

    # load responses and write to output
    sorted_ed_csv_fns = sorted([fn.strip() for fn in argv[2:]])
    questions, responses, correct = load_ed_responses(sorted_ed_csv_fns)
    sorted_emails = sorted({email for csv_fn in responses for email in responses[csv_fn]})
    with open(argv[1].strip(), 'w') as out_tsv_f:
        out_tsv = writer(out_tsv_f, delimiter='\t')
        out_tsv.writerow(["Email", "Correct"] + ['%s (%s)' % (csv_fn,q) for csv_fn in sorted(questions.keys()) for qi,q in questions[csv_fn]])
        for email in sorted_emails:
            curr_correct = list(); curr_responses = list()
            for csv_fn in sorted_ed_csv_fns:
                if email in correct[csv_fn]:
                    curr_correct += ['%s (%s)' % (csv_fn,questions[csv_fn][i]) for i in correct[csv_fn][email]]
                    curr_responses += responses[csv_fn][email]
                else:
                    curr_responses += ['']*len(questions[csv_fn])
            out_tsv.writerow([email, ','.join(curr_correct)] + curr_responses)
