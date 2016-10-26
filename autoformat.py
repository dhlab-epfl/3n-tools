# coding: utf8
#!/usr/bin/env python

from __future__ import unicode_literals
from __future__ import division
import sys

import glob, os, re
import getopt

import roman
import json
import codecs

if sys.version_info < (3,0):
	reload(sys)
	sys.setdefaultencoding('utf8')

################################################################################################################################################################

def isChapterMarker(line, expectedNumber=0, crBefore=0, crAfter=0):
	numRep = extractNumberInLine(line, expectedNumber)
	return ( line[0:8].lower() == "chapitre" or (numRep>0 and numRep<100) ) and crBefore>1;

def extractNumberInLine(line, expectedNumber):
	numRep = stringToNumber(line)
	lineParts = line.split('. ')
	if (numRep==0 and len(lineParts)>1):
		numericComponent = stringToNumber(lineParts[0])
#		if (numericComponent == expectedNumber):
		numRep = numericComponent
	return numRep

def stringToNumber(line):
	if (line[0:8].lower()=='chapitre'):
		line = line[8:].strip()
	if (line.replace('.','').isnumeric()):
		return int(line.replace('.',''))
	else:
		return (romanToNumeral(line) or romanToNumeral(line.replace('.','')))

def romanToNumeral(string):
	try:
		return roman.fromRoman(string);
	except roman.InvalidRomanNumeralError:
		return 0

def makeChapter(chapter_marker, chapter_title, tome_num, chap_title_num, chapters_lines_buff):
	return {"number":str(tome_num)+"."+str(chap_title_num), "title":(chapter_marker if chapter_title=="" else chapter_marker+" - "+chapter_title), "text":" ".join(chapters_lines_buff)}

def filterLine(line):
	line = line.strip()
	if (re.match('(\*\s*)+', line)):			# remove separator lines (« *  *  * »)
		line = ''
	if (re.match('([^\s][-‐])$', line)):		# remove hyphens
		line = ''
	return line


################################################################################################################################################################

bookfile = u''
outFolder = u''

try:
	opts, args = getopt.getopt(sys.argv[1:], "hfo", ["help", "file=", "out="])
except getopt.GetoptError as err:
    # print help information and exit:
    print(err) # will print something like "option -a not recognized"
    sys.exit(2)

for o, a in opts:
	if o in ("-h", "--help"):
		print("Options: -f[ho]")
		print(" -h   Help")
		print(" -f   File to extract")
		print(" -o   Output folder (if undefined, will write summary to console)")
		sys.exit()
	elif o in ("-f", "--file"):
		bookfile = a
	elif o in ("-o", "--out"):
		outFolder = a
	else:
		assert False, "unhandled option"

with codecs.open(bookfile, 'r', 'utf8') as f:
	chapters = []
	chapters_lines_buff = []
	chapter_title = ''
	chapter_marker = ''
	crChainCount = 0
	tomeNumber = 1
	lastSeenChapNumber = 0
	lineNumber = 0
	lines = []
	lookahead_line = ''
	for i, raw_line in enumerate(f):
		lines.append(filterLine(raw_line))
	lines.append('')
	lines_lh = zip(lines[:-1], lines[1:])
	for line, lookahead_line in lines_lh:
		lineNumber = lineNumber+1
		if (isChapterMarker(line, lastSeenChapNumber+1, crChainCount) and not isChapterMarker(lookahead_line, lastSeenChapNumber+1, crChainCount)):
			chapters.append(makeChapter(chapter_marker, chapter_title, tomeNumber, lastSeenChapNumber, chapters_lines_buff))
			newChapNumber = extractNumberInLine(line, lastSeenChapNumber+1)
			if (newChapNumber != lastSeenChapNumber+1):
				if (newChapNumber == 1):
					tomeNumber = tomeNumber+1
					lastSeenChapNumber = 1
				else:
					print('WARNING: irregular chapter number found at line '+str(lineNumber)+', expected '+str(lastSeenChapNumber+1)+' but found '+str(newChapNumber))
			else:
				lastSeenChapNumber = newChapNumber
			chapter_marker = line
			chapters_lines_buff = []
			chapter_title = ''
			crChainCount = 0
		else:
			if (len(chapters_lines_buff)==0 and chapter_title==u'' and len(line)<50):
				chapter_title = line
				crChainCount = 0
			elif (len(line)==0):
				crChainCount = crChainCount+1
				if len(chapters_lines_buff)>0:
					chapters_lines_buff.append(u"\n")
			else:
				newParagraph = ("\n" if crChainCount>1 else "")
				chapters_lines_buff.append(newParagraph+line)
				crChainCount = 0
	if (len(chapters_lines_buff)>0):
		chapters.append(makeChapter(chapter_marker, chapter_title, tomeNumber, lastSeenChapNumber, chapters_lines_buff))

	if (outFolder==u''):
		for idx, chapter in enumerate(chapters):
			print(str(idx)+".\t"+str(chapter['number'])+u".\t*"+chapter['title']+u"*\t"+chapter['text'][0:100]+u'…')
	else:
		bookfile_noext = bookfile.split(u'/')[-1].replace('.txt', '')
		f = codecs.open(outFolder+bookfile_noext+'-compact.txt', 'w+', 'utf8')
		for idx, chapter in enumerate(chapters):
			f.write(str(chapter['number'])+".\t"+chapter['title']+"\t"+chapter['text'].replace('\n', ' ')+"\n")
		f = codecs.open(outFolder+bookfile_noext+'.json', 'w+', 'utf8')
		f.write(json.dumps(chapters))

