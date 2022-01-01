# MESS: Moshiri Exam Similarity Score
**MESS** (**M**oshiri **E**xam **S**imilarity **S**core) is a scalable Python tool for detecting exam similarity from student responses. For information about the mathematical methods behind the tool, see the [Methods](../../wiki/Methods) section of the [MESS Wiki](../../wiki).

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
