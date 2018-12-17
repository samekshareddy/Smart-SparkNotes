1- run the following command to install dependencies
pip install -r requirements.txt

2- Install Spacy's english Module by running:
python -m spacy download en

3- Install textblob's corpora:
python -m textblob.download_corpora

4- download nltks's wordnet and stopwords:
nltk.download('wordnet')
nltk.download('stopwords')

5- To start the server run,
python UI.py
Navigate to the link specified in the output of the above program