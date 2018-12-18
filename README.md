# Smart Notes

Summarization

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

THE TOOL WORKS FOR MAC AND LINUX ONLY NOW, WE ARE WORKING ON GETTING IT RUNNING FOR WINDOWS

The prerequisites are mentioned in the requirements.txt. To install the packages run:

pip install -r requirements.txt


Install Spacy's english Module by running:

python -m spacy download en


Install textblob's corpora:  


python -m textblob.download_corpora. 

Download nltks's wordnet and stopwords:  


nltk.download('wordnet'). 

nltk.download('stopwords'). 

nltk.download(' averaged_perceptron_tagger'). 


### Running the tool

To start the server run,  

python UI.py. 


Navigate to the link specified in the output of the above program. 


To get the full summary of the paper, enable full summary checkbox.  
To get summary of specfic sections, mention comma seperated headings in the text book.  
Query can be specified in its respective text box. 
Number of sentences for the summary should be specified. 



## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


