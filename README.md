# 3n-tools
## A set of simple novel authoring and analysis tools
-----------------------------------------------------

##Installation - Python dependencies

###matplotlib

```
apt-get install python-matplotlib
```

###scipy

```
apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
```

###hunspell

```
~$	apt-get update
~$	apt-get install python2.7-dev
~$	apt-get install libhunspell-dev
~$	pip install hunspell
```

###treetagger3

http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/

###nltk (needed for treetagger3)

```
~$	pip install -U nltk
~$	pip install treetaggerwrapper
```

###treetagger-python

https://github.com/miotto/treetagger-python

Update `os.environ["TREETAGGER_HOME"]` in characterStats.py

The `./cache folder` needs to be writeable.

###roman

```
~$	pip install roman
```

https://pypi.python.org/pypi/roman

##About File Formats

For historical reasons, characterStats.py needs books to be plain text, formatted as one chapter per line (all the text of the chapter is on the same line, preceded by the chapter number). To format a book as "1 chapter per line", a handy conversion script is provided: 

```
python autoformat.py --file="books/madamebovary.txt" --out="books/"
```

=> will output the necessary `madamebovary-compact.txt` file


##Examples

###Graphical Representations

The most direct use of characterStats is to output graphical visualizations of the story:

```
python characterStats.py --file="books/madamebovary-compact.txt" -g
```

For some books, better results can be achieved by also looking up wikipedia pages for disambiguation:

```
python characterStats.py --file="books/madamebovary-compact.txt" -g --mwclient="fr.wikipedia.org"
```

###Setting the minimal number of occurences

By default, most significant entities are selected according to an heuristic. You can specify a fixed threshold with the -c parameter. The following example will visualize only entities quoted more than 20 times:

```
python characterStats.py --file="books/madamebovary-compact.txt" -c20 -g
```

###Extracting Named Entities

Verbose mode will output the classification results to console, where each row contains the name of the entity, assigned classification, confidence grade and number of citations:

```
python characterStats.py --file="books/madamebovary-compact.txt" -v

Emma	character	0.769230769231	354
Charles	character	0.840909090909	297
Rouen	place	0.625	59
[…]
```

Alternatively, summarized classification results (without confidence marks) can be output in JSON format:

```
python characterStats.py --file="books/madamebovary-compact.txt" -a

{
	"classes": {
		"place": ["Paris", "Rouen", "Tostes" […] "Croix"],
		"character": ["Lempereur" […] "Charles", "Emma"]
	},
	"substitutions": {"BalzacGeorge": "Balzac George", […]}
}
```


###Debug Mode


Debug mode will explain in details the classification process and output possible warnings:

```
python characterStats.py --file="books/madamebovary-compact.txt" -d
```


###Benchmark Example

The included `madamebovary-compact.corr` file shows a simple benchmark set that can be used to evaluate the quality of the classification.

```
python characterStats.py --file="books/madamebovary-compact.txt" -b
```
