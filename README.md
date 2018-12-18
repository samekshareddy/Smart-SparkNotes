# Smart Notes

Summarization

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

THE TOOL WORKS FOR MAC AND LINUX ONLY NOW, WE ARE WORKING ON GETTING IT RUNNING FOR WINDOWS

The prerequisites are mentioned in the requirements.txt. 

#### To install the packages run:

pip install -r requirements.txt


#### Install Spacy's english Module by running:

python -m spacy download en


#### Install textblob's corpora:  

python -m textblob.download_corpora. 

#### Install gensim

pip install --upgrade gensim

#### Download nltks's wordnet and stopwords:  

nltk.download('wordnet'). 

nltk.download('stopwords'). 

nltk.download(' averaged_perceptron_tagger'). 

### Load Google word2vec model

1. Download the model (very large file - 3.5 gb) from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

2. Place this model file in the same folder as initialize.py

3. Load the model with python initialize.py

### Running the tool

To start the server run,  

1. python UI.py. 


2. Navigate to the link specified in the output of the above program. 

#### Notes

1. Once clicking on the upload button, please wait for the file to get uploaded. There is no progess button since the UI is still in progress.  

2. To get the full summary of the paper, enable full summary checkbox.  
3. To get summary of specfic sections, mention comma seperated headings in the text box.  
4. Query can be specified in its respective text box. 
5. Number of sentences for the summary should be specified.  
6. The summary takes a while (100-140 sec) to get generated.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details


