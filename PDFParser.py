from PyPDF2 import PdfFileReader
import subprocess
from bs4 import BeautifulSoup
import re
from nltk.corpus import wordnet
from nltk.corpus import words
import json
from nltk.corpus import wordnet
from smartnotes import SmartNote

class PDFParser:

	variations = {}

	htmlContent = None

	info = None

	elements = None

	headings = []


	def convert_html(self,path):
		self.info = self.get_info(path)
		htmlPath = path.split('.')[0] + ".html"
		args2 = ['pdf2txt.py','-t', 'html', path]

		output = ""
		proc = subprocess.Popen(args2,stdout=subprocess.PIPE)

		for line in proc.stdout:
			output += line.decode("utf-8")

		self.htmlContent = output
		
		self.elements = self.get_content(self.htmlContent)

		self.headings = self.getAllHeadings(self.elements)
	
		return output

	def get_info(self,path):
	    with open(path, 'rb') as f:
	        pdf = PdfFileReader(f)
	        info = pdf.getDocumentInfo()
	        number_of_pages = pdf.getNumPages()
	 	 
	    author = info.author
	    creator = info.creator
	    producer = info.producer
	    subject = info.subject
	    title = info.title

	    return info;

	def preprocessContent(self,content):
		content = content.replace('\n', ' ').replace('\r', '')

		references = re.findall(re.compile("\[[\d,]+\]"), content)

		emails = re.findall(re.compile("\s.+@.+\s"), content)
		
		contWords = re.findall(re.compile("[a-zA-Z]+-\s[a-zA-Z]+[,\)]|[a-zA-Z]+-\s[a-zA-Z]+\s"), content)

		# d = enchant.Dict("en_US")

		for word in contWords:
			word = word[0:-1]
			newWord = word.replace("- ","")

			# if d.check(newWord.strip()):
			if len(wordnet.synsets(newWord.strip())) != 0:
				content = content.replace(word,newWord)


		for email in emails:
			# print(email)
			content = content.replace(email,"")

		for reference in references:
			content = content.replace(reference,"")
		
		
		return content

	def get_content(self,htmlContent):
		soup = BeautifulSoup(htmlContent) 

		pattFontSize = re.compile("font-size:(\d+)")
		pattFontFamily = re.compile("font-family:(.+;)")

		ele = [(tag.text.strip(), pattFontSize.search(tag["style"]).group(1),pattFontFamily.search(tag["style"]).group(1)) for tag in soup.select("[style*=font-size]")]

		return ele


	def getAllHeadings(self,ele):
		headings = []

		fontSize = None
		index = None
		fontFamily = None

		for data in range(len(ele)):
			text = ele[data][0]
			size = ele[data][1]
			font = ele[data][2]

			# print(text)

			if "introduction" in text.lower():
				fontSize = size
				index = data
				fontFamily = font

		
		for data in range(len(ele)):
			text = ele[data][0]
			size = ele[data][1]
			font = ele[data][2]

			if size == fontSize and font == fontFamily:
				headings.append(text)

		return headings



	def get_headings(self,ele,heading,isFull,nextHeading = False, nextHeadingText = None):
		
		fontSize = None
		index = None
		fontFamily = None

		if isFull:
			content = ""
			for data in range(len(ele)):
				content += " " + ele[data][0];

			return content
			

		for data in range(len(ele)):
			text = ele[data][0]
			size = ele[data][1]
			font = ele[data][2]

			# print(text)

			if heading.lower() in text.lower():
				fontSize = size
				index = data
				fontFamily = font

			if heading in self.variations:
				for h in self.variations[heading]:
					if h.lower() in text.lower():
						fontSize = size
						index = data
						fontFamily = font


		content = ""
		contentWords = []
		first = True

		if index == None:
			return ""

		for data in range(index,len(ele)):
			
			if first:
				first = False
				continue
				
			elif fontSize == ele[data][1] and fontFamily == ele[data][2] and content != "":
				break;

			if nextHeading:
				if nextHeadingText in ele[data][0]:
					break

			# d = enchant.Dict("en_US")

			
			if len(contentWords) == 0:
				content = ele[data][0]

			else:
				# word = contentWords[len(contentWords) - 1] + ele[data][0].split()[0]

				
				# # if d.check(word.strip()):
				# if len(wordnet.synsets(word.strip())) != 0:
				# 	content += ele[data][0]

				# else:
				content += " " + ele[data][0];

			
			contentWords.extend(content.split())



		return content

	def readPDF(self, request):

		start = request['start']

		headings = start.split(',')
		
		isFull = request['isFull']

		self.variations['abstract'] = [ "a b s t r a c t" ]
		
		isEnd = False

		sm = SmartNote()

		

		# elements = self.get_content(self.htmlContent)
		contentDict = {}

		if isFull:
			headings = self.getAllHeadings(self.elements)
			print(headings)
		
		for heading in headings:
			contentDict[heading] = self.preprocessContent(self.get_headings(self.elements,heading,False,isEnd,""))
		

		print(self.info.title)
		print(contentDict)
		# paper =  json.dumps(contentDict)

		summary = {}

		summary["content"] = sm.getSummary(contentDict,self.info.title,request['length'],request['query'])

		return json.dumps(summary)




