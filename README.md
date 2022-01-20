# MESS: Moshiri Exam Similarity Score
**MESS** (**M**oshiri **E**xam **S**imilarity **S**core; pun on "MOSS") is a scalable Python tool for detecting exam similarity from student responses. For information about the mathematical methods behind the tool, see the [Methods](../../wiki/Methods) section of the [MESS Wiki](../../wiki).

## Installation
MESS is written in Python 3 and depends on [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), and [seaborn](https://seaborn.pydata.org/), which can be installed using `pip3`:

```bash
pip3 install numpy scipy seaborn
```

You can simply download [MESS.py](MESS.py) to your machine and make it executable:

```bash
wget "https://raw.githubusercontent.com/niemasd/MESS/main/MESS.py"
chmod a+x MESS.py
sudo mv MESS.py /usr/local/bin/MESS.py # optional step to install globally
```
## Usage
A comprehensive list of MESS arguments can be found below, but we highly recommend following the [Tutorial](../../wiki/Tutorial) section of the [MESS Wiki](../../wiki).

```
usage: MESS.py [-h] -i INPUT -ot OUTPUT_TSV -op OUTPUT_PDF
               [--ignore_case] [-c CORRECTION]
               [-rm REG_MIN] [-rM REG_MAX] [-rd REG_XDELTA]
               [-kc KDE_COLOR] [-kl KDE_LINESTYLE] [-kw KDE_LINEWIDTH]
               [-rc REG_COLOR] [-rl REG_LINESTYLE] [-rw REG_LINEWIDTH]
               [-t TITLE] [-xl XLABEL] [-xm XMIN] [-xM XMAX] [-yl YLABEL] [-ym YMIN] [-yM YMAX] [--no_ylog]

optional arguments:
  -h, --help                                         show this help message and exit
  -i INPUT, --input INPUT                            Input Exam Responses (TSV) (default: None)
  -ot OUTPUT_TSV, --output_tsv OUTPUT_TSV            Output MESS Spreadsheet (TSV) (default: None)
  -op OUTPUT_PDF, --output_pdf OUTPUT_PDF            Output MESS Distribution (PDF) (default: None)
  --ignore_case                                      Ignore Case in Student Responses (default: False)
  -c CORRECTION, --correction CORRECTION             Multiple Hypothesis Test Correction (options: benjamini_hochberg, bonferroni, none) (default: benjamini_hochberg)
  -rm REG_MIN, --reg_min REG_MIN                     Minimum MESS for Regression (default: None)
  -rM REG_MAX, --reg_max REG_MAX                     Maximum MESS for Regression (default: None)
  -rd REG_XDELTA, --reg_xdelta REG_XDELTA            X Delta for Regression (default: 0.0001)
  -kc KDE_COLOR, --kde_color KDE_COLOR               KDE Color (default: black)
  -kl KDE_LINESTYLE, --kde_linestyle KDE_LINESTYLE   KDE Linestyle (default: --)
  -kw KDE_LINEWIDTH, --kde_linewidth KDE_LINEWIDTH   KDE Line Width (default: 0.75)
  -rc REG_COLOR, --reg_color REG_COLOR               Regression Color (default: black)
  -rl REG_LINESTYLE, --reg_linestyle REG_LINESTYLE   Regression Linestyle (default: -)
  -rw REG_LINEWIDTH, --reg_linewidth REG_LINEWIDTH   Regression Line Width (default: None)
  -t TITLE, --title TITLE                            Figure Title (default: MESS Distribution)
  -xl XLABEL, --xlabel XLABEL                        Figure X-Axis Label (default: MESS Score)
  -xm XMIN, --xmin XMIN                              Figure Minimum X (default: 0)
  -xM XMAX, --xmax XMAX                              Figure Maximum X (default: None)
  -yl YLABEL, --ylabel YLABEL                        Figure Y-Axis Label (default: Frequency)
  -ym YMIN, --ymin YMIN                              Figure Minimum Y (default: None)
  -yM YMAX, --ymax YMAX                              Figure Maximum Y (default: None)
  --no_ylog                                          Don't Plot Y-Axis in Log-Scale (default: False)
```

# Citing MESS
A manuscript has been accepted to CCSC-SW 2022, but while it's in press, please cite this GitHub repository.
