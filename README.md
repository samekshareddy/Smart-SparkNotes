# Smart Notes

Summarization

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

The prerequisites are mentioned in the requirements.txt. To install the packages run:

pip install -r requirements.txt


Install Spacy's english Module by running:

python -m spacy download en


Install textblob's corpora: < br />

python -m textblob.download_corpora < br />

Download nltks's wordnet and stopwords: < br />

nltk.download('wordnet') < br />
nltk.download('stopwords') < br />
nltk.download(' averaged_perceptron_tagger') < br />

### Running the tool

To start the server run, < br /> < br />
python UI.py

Navigate to the link specified in the output of the above program

To get the full summary of the paper, enable full summary checkbox. < br />
To get summary of specfic sections, mention comma seperated headings in the text book. < br />
Query can be specified in its respective text box < br />
Number of sentences for the summary should be specified < br />


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


