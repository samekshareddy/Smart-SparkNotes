from PyPDF2 import PdfFileReader
import subprocess
from bs4 import BeautifulSoup
import re
from nltk.corpus import wordnet
from nltk.corpus import words
import enchant
import json

class PDFParser:

	variations = {}

	htmlContent = None

	def convert_html(self,path):
		htmlPath = path.split('.')[0] + ".html"
		args2 = ['pdf2txt.py','-t', 'html', path]

		output = ""
		proc = subprocess.Popen(args2,stdout=subprocess.PIPE)

		for line in proc.stdout:
			output += line.decode("utf-8")

		self.htmlContent = output
	
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

	def preprocessContent(self,content):
		content = content.replace('\n', ' ').replace('\r', '')

		references = re.findall(re.compile("\[[\d,]+\]"), content)
		
		contWords = re.findall(re.compile("[a-zA-Z]+-\s[a-zA-Z]+[,\)]|[a-zA-Z]+-\s[a-zA-Z]+\s"), content)

		d = enchant.Dict("en_US")

		for word in contWords:
			word = word[0:-1]
			newWord = word.replace("- ","")

			if d.check(newWord.strip()):
				content = content.replace(word,newWord)
		
		
		return content

	def get_content(self,htmlContent):
		soup = BeautifulSoup(htmlContent) 

		pattFontSize = re.compile("font-size:(\d+)")
		pattFontFamily = re.compile("font-family:(.+;)")

		ele = [(tag.text.strip(), pattFontSize.search(tag["style"]).group(1),pattFontFamily.search(tag["style"]).group(1)) for tag in soup.select("[style*=font-size]")]

		return ele

	def get_headings(self,ele,heading,isFull,nextHeading = False, nextHeadingText = None):

		print(len(ele))
		
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

			d = enchant.Dict("en_US")

			
			if len(contentWords) == 0:
				content = ele[data][0]

			else:
				word = contentWords[len(contentWords) - 1] + ele[data][0].split()[0]

				
				if d.check(word.strip()):
					content += ele[data][0]

				else:
					content += " " + ele[data][0];

			
			contentWords.extend(content.split())



		return content

	def readPDF(self, request):
		print(request)
		headings = request['start']
		
		isFull = request['isFull']

		self.variations['abstract'] = [ "a b s t r a c t" ]
		
		isFull = False
		isEnd = False

		if headings == "":
			isFull = True
		
		# if end == "":
		# 	isEnd = False


		
		elements = self.get_content(self.htmlContent)
		contentDict = {}

		for heading in headings.split(','):
			contentDict[heading] = self.preprocessContent(self.get_headings(elements,heading,isFull,isEnd,""))
		

		return json.dumps(contentDict)




