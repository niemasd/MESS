#! /usr/bin/env python3
'''
MESS: Moshiri Exam Similarity Score
'''

# imports
from csv import reader, writer
from datetime import datetime
from math import log
from numpy import arange, histogram
from os.path import isfile
from scipy.stats import expon, gaussian_kde, linregress
from seaborn import histplot, kdeplot
from statistics import median
from sys import argv, stderr, stdout
from warnings import warn
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# constants
VERSION = '1.0.9'

# no correction
def qvalues_nocorrection(pvalues):
    return list(pvalues)

# bonferroni correction
def qvalues_bonferroni(pvalues):
    len_p = len(pvalues)
    return [min(1, p*len_p) for p in pvalues]

# benjamini-hochberg correction
def qvalues_benjamini_hochberg(pvalues):
    len_p = len(pvalues); qvalues = [None for _ in range(len_p)]
    for rank, pair in enumerate(sorted((p,ind) for ind,p in enumerate(pvalues))):
        p, ind = pair; qvalues[ind] = min(1, p*len_p/(rank+1))
    return qvalues

# constants about the correction techniques
CORRECTION = {
    'bonferroni': {
        'name': "Bonferroni",
        'func': qvalues_bonferroni,
    },
    'benjamini_hochberg': {
        'name': "Benjamini-Hochberg",
        'func': qvalues_benjamini_hochberg,
    },
    'none': {
        'name': "No Correction",
        'func': qvalues_nocorrection,
    },
}

# return the current time as a string
def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# print to stdout log (prefixed by current time)
def print_log(s='', end='\n'):
    tmp = "[%s] %s" % (get_time(), s); print(tmp, end=end); stdout.flush()

# throw error
def error(message, prefix="ERROR: ", out_file=stderr):
    print("%s%s" % (prefix, message), file=out_file); exit(1)

# load input data
def load_input_data(in_tsv_fn, ignore_case=False):
    # prepare for loading data
    questions = list() # questions = list containing the question labels in the order they appear in the header
    question_to_ind = dict() # question_to_ind[question] = index in `questions` (`q_ind`) for `question`
    responses = dict() # responses[student][q_ind] = response from `student` for `q_ind`
    correct = dict() # correct[student] = set containing the question label indices (`q_ind`) `student` got correct

    # load data and return
    with open(in_tsv_fn) as infile:
        for row_num, row in enumerate(reader(infile, delimiter='\t')):
            # load question labels from header
            if row_num == 0:
                for q_ind, q_orig in enumerate(row[2:]):
                    q = q_orig.strip()
                    if q in question_to_ind:
                        error("Duplicate question label: %s" % q)
                    questions.append(q); question_to_ind[q] = q_ind
                continue

            # load current student's correct questions
            curr_student = row[0].strip(); curr_correct = set()
            if len(row[1].strip()) != 0:
                for q_orig in row[1].split(','):
                    q = q_orig.strip()
                    if q in curr_correct:
                        error("Duplicate correct question: %s for student %s" % (q, curr_student))
                    if q not in question_to_ind:
                        error("Student correct question label not found in header row: question '%s' for student '%s'" % (q, curr_student))
                    curr_correct.add(question_to_ind[q])
            correct[curr_student] = curr_correct

            # load current student's responses
            if ignore_case:
                curr_responses = [v.strip().lower() for v in row[2:]] # lowercase responses to ignore case
            else:
                curr_responses = [v.strip() for v in row[2:]]
            if len(curr_responses) > len(questions):
                error("Row has more responses than there are questions in the header: %s" % curr_student)
            elif len(curr_responses) < len(questions):
                curr_responses += ['']*(len(questions)-len(curr_responses))
            responses[curr_student] = curr_responses
    return questions, responses, correct

# compute similarity scores
def compute_mess(questions, responses, correct, ignore_case=False):
    # prepare helpful variables
    sorted_students = sorted(responses.keys())
    num_students = len(sorted_students)
    num_questions = len(questions)

    # count the number of unique responses to each question
    response_count = [{'correct':dict(),'incorrect':dict()} for _ in range(num_questions)] # response_count[q_ind]['correct'/'incorrect'][response] = number of students who submitted `response` for `questions[q_ind]`
    for student in responses:
        for q_ind, response in enumerate(responses[student]):
            if q_ind in correct[student]:
                curr = response_count[q_ind]['correct']
            else:
                curr = response_count[q_ind]['incorrect']
            if response in curr:
                curr[response] += 1
            else:
                curr[response] = 1

    # count the number of students who got each question correct and incorrect
    correct_count = [0 for _ in range(num_questions)]
    for student in correct:
        for q_ind in correct[student]:
            correct_count[q_ind] += 1

    # compute MESS and proportion identical for all pairs of students
    mess = list() # mess = list of (proportion identical, MESS score, student1, student2, red_count, yellow_count, green_count) tuples; red = same wrong answer, yellow = diff wrong answers, green = only 1 student missed
    for student1_ind in range(num_students-1):
        student1 = sorted_students[student1_ind]; responses1 = responses[student1]
        for student2_ind in range(student1_ind+1, num_students):
            student2 = sorted_students[student2_ind]; responses2 = responses[student2]
            score = 0; prop_identical = 0.; red_count = 0; yellow_count = 0; green_count = 0
            for q_ind in range(num_questions):
                rs1 = responses1[q_ind]; rs2 = responses2[q_ind]
                if rs1 == rs2: # both students put identical answers (regardless of right or wrong)
                    prop_identical += 1
                    if q_ind not in correct[student1] and len(rs1) != 0:
                        # both students put the same non-empty wrong answer
                        red_count += 1
                        num_wrong = sum(response_count[q_ind]['incorrect'].values())
                        num_diff_wrong = num_wrong - response_count[q_ind]['incorrect'][rs1]
                        if num_diff_wrong < 0:
                            error_message = "Number of different wrong answers was negative (%s): question '%s' for students '%s' and '%s' (%d correct, %d incorrect)" % (num_diff_wrong, questions[q_ind], student1, student2, correct_count[q_ind], num_wrong)
                            if ignore_case:
                                error("%s\nPerhaps '--ignore_case' is not valid for this question (e.g. correctness is case-dependent)?" % error_message)
                            else:
                                error(error_message)
                        score += (float(num_diff_wrong)/num_wrong) # prop students who put a different wrong answer
                elif q_ind not in correct[student1] and q_ind not in correct[student2]: # both students got it wrong, but they put different wrong answers
                    yellow_count += 1
                elif q_ind not in correct[student1] or q_ind not in correct[student2]: # only 1 student got it wrong
                    green_count += 1
            prop_identical /= num_questions; score /= num_questions # normalize by number of questions
            mess.append((score, prop_identical, student1, student2, red_count, yellow_count, green_count))
    return mess

# perform regression on log-scale MESS distribution
def regress_mess(mess_scores, reg_min, reg_max, reg_xdelta):
    kde = gaussian_kde(mess_scores)
    X = arange(reg_min, reg_max, reg_xdelta)
    Y = kde.logpdf(X)
    line = linregress(X,Y) # y = ln(L) - Lx, where L = rate parameter (lambda) of Exponential distribution
    rate = -1 * line.slope; scale = 1. / rate
    loc = (log(rate) - line.intercept)/line.slope
    return rate, scale, loc

# plot MESS distribution + regression
def plot_mess(mess_scores, scale, loc, xdelta, min_mess_test=None, kde_color='black', kde_linestyle='--', kde_linewidth=0.75, reg_color='black', reg_linestyle='-', reg_linewidth=None, title=None, xlabel=None, xmin=0, xmax=None, ylabel=None, ymin=None, ymax=None, ylog=True, show_hist=True, show_num_tests=True):
    fig, ax = plt.subplots()
    if show_hist:
        histplot(mess_scores, stat='density', fill=False)
    kdeplot(mess_scores, color=kde_color, linestyle=kde_linestyle, linewidth=kde_linewidth)
    if xmax is None:
        xmax = ax.get_xlim()[1]
    Xplot = arange(loc+xdelta, xmax, xdelta)
    Yplot = expon.pdf(Xplot, loc=loc, scale=scale)
    plt.plot(Xplot, Yplot, color=reg_color, linestyle=reg_linestyle)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if ylog:
        ax.set_yscale('log')
    if ymin is None:
        ymin = ax.get_ylim()[0]
    if ymax is None:
        ymax = ax.get_ylim()[1]
    if min_mess_test is not None and show_num_tests:
        plt.plot([min_mess_test,min_mess_test], [ymin,ymax], color='red', linestyle=':')
    plt.xlim(xmin=xmin, xmax=xmax); plt.ylim(ymin=ymin, ymax=ymax)
    return fig, ax

# compute theoretical p-values
def compute_pvals(mess_scores, scale, loc):
    unique_pvals = {s:1.-expon.cdf(s,loc=loc,scale=scale) for s in set(mess_scores)}
    return [unique_pvals[s] for s in mess_scores]

# write output TSV
def write_mess_output(output_tsv_fn, mess, p_values, q_values, rate, loc, correction):
    with open(output_tsv_fn, 'w') as out_tsv_f:
        out_tsv = writer(out_tsv_f, delimiter='\t')
        out_tsv.writerow(["Student 1", "Student 2", "MESS", "Proportion Identical", "p-value (rate=%s, loc=%s)" % (rate,loc), "q-value (%d tests; MESS >= %s; correction: %s)" % (args.num_tests, min_mess_test, CORRECTION[correction]['name']), "Red (same wrong answer)", "Yellow (different wrong answers)", "Green (only 1 student missed)"])
        for i in range(len(mess)):
            m, ident, u, v, r, y, g = mess[i]; p = p_values[i]
            if i < len(q_values):
                q = q_values[i]
            else:
                q = "N/A"
            out_tsv.writerow([u, v, m, ident, p, q, r, y, g])

# find number of significance tests to perform
def find_num_tests(mess_scores):
    hist, bin_edges = histogram(mess_scores, bins='auto')
    med = median(mess_scores)
    for i in range(len(hist)):
        if hist[i] == 0 and bin_edges[i] > med:
            min_mess_test = bin_edges[i]; break
    return min_mess_test, len([v for v in mess_scores if v > min_mess_test])

# parse user args
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True, type=str, help="Input Exam Responses (TSV)")
    parser.add_argument('-ot', '--output_tsv', required=True, type=str, help="Output MESS Spreadsheet (TSV)")
    parser.add_argument('-op', '--output_pdf', required=True, type=str, help="Output MESS Distribution (PDF)")
    parser.add_argument('--ignore_case', action='store_true', help="Ignore Case in Student Responses")
    parser.add_argument('-nt', '--num_tests', required=False, type=int, default=None, help="Number of Significance Tests to Perform")
    parser.add_argument('-c', '--correction', required=False, type=str, default='benjamini_hochberg', help="Multiple Hypothesis Test Correction (options: %s)" % ', '.join(sorted(CORRECTION.keys())))
    parser.add_argument('-rm', '--reg_min', required=False, type=float, default=None, help="Minimum MESS for Regression")
    parser.add_argument('-rM', '--reg_max', required=False, type=float, default=None, help="Maximum MESS for Regression")
    parser.add_argument('-rd', '--reg_xdelta', required=False, type=float, default=0.0001, help="X Delta for Regression")
    parser.add_argument('-kc', '--kde_color', required=False, type=str, default='black', help="KDE Color")
    parser.add_argument('-kl', '--kde_linestyle', required=False, type=str, default='--', help="KDE Linestyle")
    parser.add_argument('-kw', '--kde_linewidth', required=False, type=float, default=0.75, help="KDE Line Width")
    parser.add_argument('-rc', '--reg_color', required=False, type=str, default='black', help="Regression Color")
    parser.add_argument('-rl', '--reg_linestyle', required=False, type=str, default='-', help="Regression Linestyle")
    parser.add_argument('-rw', '--reg_linewidth', required=False, type=str, default=None, help="Regression Line Width")
    parser.add_argument('-sh', '--show_hist', action='store_true', help="Show Histogram")
    parser.add_argument('-st', '--show_num_tests', action='store_true', help="Show Number of Significance Tests Performed")
    parser.add_argument('-t', '--title', required=False, type=str, default="MESS Distribution", help="Figure Title")
    parser.add_argument('-xl', '--xlabel', required=False, type=str, default="MESS Score", help="Figure X-Axis Label")
    parser.add_argument('-xm', '--xmin', required=False, type=float, default=0, help="Figure Minimum X")
    parser.add_argument('-xM', '--xmax', required=False, type=float, default=None, help="Figure Maximum X")
    parser.add_argument('-yl', '--ylabel', required=False, type=str, default="Frequency", help="Figure Y-Axis Label")
    parser.add_argument('-ym', '--ymin', required=False, type=float, default=None, help="Figure Minimum Y")
    parser.add_argument('-yM', '--ymax', required=False, type=float, default=None, help="Figure Maximum Y")
    parser.add_argument('--no_ylog', action='store_true', help="Don't Plot Y-Axis in Log-Scale")
    return parser.parse_args()

# main content
if __name__ == "__main__":
    # parse and check user args
    args = parse_args()
    if not isfile(args.input):
        error("Input file not found: %s" % args.input)
    for fn in [args.output_tsv, args.output_pdf]:
        if isfile(fn):
            error("Output file exists: %s" % fn)
    args.correction = args.correction.lower()
    if args.num_tests is not None and args.num_tests < 1:
        error("Number of hypothesis tests needs to be positive: %s" % args.num_tests)
    if args.correction not in CORRECTION:
        error("Invalid multiple hypothesis test correction: %s\nOptions: %s" % (args.correction, ', '.join(sorted(CORRECTION.keys()))))
    if args.reg_min is not None:
        if args.reg_min <= 0:
            error("reg_min must be positive: %s" % args.reg_min)
        if args.reg_max is not None:
            if args.reg_max <= args.reg_min:
                error("reg_max must be greater than reg_min. reg_min: %s and reg_max: %s" % (args.reg_min, args.reg_max))
            if args.reg_xdelta >= (args.reg_max - args.reg_min):
                error("reg_xdelta must be smaller than reg_max - reg_min. reg_xdelta: %s and reg_max: %s and reg_min: %s" % (args.reg_xdelta, args.reg_max, args.reg_min))

    # print run information
    print_log("Running MESS v%s (Niema Moshiri 2021)" % VERSION)
    print_log("MESS Command: %s" % ' '.join(argv))

    # load input data
    print_log("Loading exam responses from input file: %s" % args.input)
    questions, responses, correct = load_input_data(args.input, ignore_case=args.ignore_case)
    print_log("Successfully loaded responses from %d students for %d exam questions" % (len(responses), len(questions)))

    # compute MESS scores
    print_log("Computing MESS scores...")
    mess = compute_mess(questions, responses, correct, args.ignore_case)
    print_log("Finished computing %d pairwise MESS scores" % len(mess))

    # process MESS scores
    print_log("Processing MESS scores...")
    mess.sort(reverse=True) # sort in descending order of MESS
    mess_scores = [m for m,ident,u,v,r,y,g in mess]
    min_mess = min(v for v in mess_scores if v > 0); max_mess = max(mess_scores)
    if args.reg_min is None:
        args.reg_min = min_mess
    else:
        args.reg_min = max(args.reg_min, min_mess)
    if args.reg_max is None:
        args.reg_max = max_mess
    else:
        args.reg_max = min(args.reg_max, max_mess)
    print_log("Finished processing MESS scores. [min, max] = [%s, %s]" % (min_mess, max_mess))

    # perform regression
    print_log("Performing linear regression from log-scale MESS distribution in MESS range [%s, %s]..." % (args.reg_min, args.reg_max))
    rate, scale, loc = regress_mess(mess_scores, reg_min=args.reg_min, reg_max=args.reg_max, reg_xdelta=args.reg_xdelta)
    print_log("Finished performing linear regression. Best fit exponential: rate = %s (scale = 1/rate = %s) and loc = %s" % (rate, scale, loc))

    # compute theoretical p-values
    print_log("Computing theoretical p-values...")
    p_values = compute_pvals(mess_scores, scale, loc)
    print_log("Finished computing theoretical p-values")

    # perform multiple hypothesis test correction
    if args.num_tests is None:
        min_mess_test, args.num_tests = find_num_tests(mess_scores)
    else:
        min_mess_test = mess_scores[args.num_tests-1]
    print_log("Performing %d significance tests... Minimum MESS: %s" % (args.num_tests, min_mess_test))
    print_log("Multiple hypothesis test correction method: %s" % CORRECTION[args.correction]['name'])
    q_values = CORRECTION[args.correction]['func'](p_values[:args.num_tests])
    print_log("Finished computing q-values (corrected p-values)")

    # write output TSV
    print_log("Writing output MESS TSV...")
    write_mess_output(args.output_tsv, mess, p_values, q_values, rate, loc, args.correction)
    print_log("Finished writing output MESS TSV: %s" % args.output_tsv)

    # plot MESS distribution + regression
    print_log("Plotting MESS distribution and regression...")
    fig, ax = plot_mess(mess_scores, scale, loc, args.reg_xdelta, min_mess_test=min_mess_test, kde_color=args.kde_color, kde_linestyle=args.kde_linestyle, kde_linewidth=args.kde_linewidth, reg_color=args.reg_color, reg_linestyle=args.reg_linestyle, reg_linewidth=args.reg_linewidth, title=args.title, xlabel=args.xlabel, xmin=args.xmin, xmax=args.xmax, ylabel=args.ylabel, ymin=args.ymin, ymax=args.ymax, ylog=(not args.no_ylog), show_hist=args.show_hist, show_num_tests=args.show_num_tests)
    fig.savefig(args.output_pdf, format='pdf', bbox_inches='tight'); plt.close(fig)
    print_log("MESS distribution and regression figure written to PDF: %s" % args.output_pdf)
