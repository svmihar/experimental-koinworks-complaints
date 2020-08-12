# Scraping 
## twitter
- [x] koinworks
- [x] koinwork

## news site?
- kompas? 
- google?
dunno, will do if feeling cute lol
[1](https://swa.co.id/swa/trends/koinworks-catat-pertumbuhan-30-pasca-pelonggaran-psbb)

## preprocessing with texthero
- [x] remove brackets 
- [x] remove diacritic
- [x] remove punctuation
- [x] remove numbers? 
- [x] remove indo stopwords
- [ ] drop duplicate tweets: 
	- promo2 gak jelas
	- referal code -> biasa satu user bisa share beberapa kali

## EDA 
- [x] top 100 most common unigram
- [ ] top 100 most common bigram
- [ ] top 100 most common trigram
- [x] wordcloud
- [ ] maybe topic modelling with LDA
- visualize
	- ~~tfidf~~
	- kmeans
	- word2vec(?)

## labelling
- search tweet with a definite "keluhan", then use cosine similarity to search similar ones, then label it too as keluhan
- do as above but instead of keluhan, search for the "good thigs"
- do as above but search, the non essential tweets (promo, etc)


### extras
- dari search twitter sempet peak di 263 tweet di 04-02-2020 dan 09-01-2020
- dari kmeans, langsung kepisah dengan cantik 3 label
