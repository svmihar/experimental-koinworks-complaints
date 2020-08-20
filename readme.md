# Scraping tweet tentang koinworks
we got em boys. keluhan classifier. classifiy which tweet are "keluhan terhadap telat bayar ke koinworks"

## approach
### finding the keluhan tweets
- scrape the tweets
- pretrain embeddings using flair
- topic model using dbscan on flair's embedding
- find a topic that describes a keluhan well
	- from wordings
	- random sample
- re label the assumed keluhan tweet
	- involves delete the tweets that aren't related to keluhan
- after finding a good keluhan dataset find similar tweets (which aren't not the in the keluhan dataset)
	- making sure all tweets are keluhan and not keluhan

### methods
all these method to find the keluhan tweet are thematically related.
- scraping: twint
- topic model: ktrain's get_document_topic
- similar texts: cosine similarity (on tfidf trained with ktrain's)
	k train is using the one class classification (svm)

## twitter search keywords
- [x] koinworks
- [x] koinwork


## pipeline
- [ ] make one script pipeline


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
- [x] drop duplicate tweets:
	- promo2 gak jelas -> biasanya bot
	- referal code -> biasa satu user bisa share beberapa kali
- [x] drop koinwork's own tweets

## EDA
- [x] top 100 most common unigram
- [x] top 100 most common bigram
- [x] top 100 most common trigram
- [x] wordcloud
- [x] maybe topic modelling with LDA
- [ ] distribusi kata yang merupakan keluhan
- visualize
	- ~~tfidf~~
	- kmeans
	- flair (pca-ed lol)

## labelling
- search tweet with a definite "keluhan", then use cosine similarity to search similar ones, then label it too as keluhan
cek di `koinworks_labeled_lda.csv`

mostlikely keluhan keywords:
['telat', ]

- do as above but instead of keluhan, search for the "good thigs"
- do as above but search, the non essential tweets (promo, etc)

## frontend
- dashboard:
	- daily keluhan berapa
	- top keywords keluhan
- search engine
	- bisa tau kasus mana yang mirip dengan yang dicari
		- ini ngelist username, tweet sama tanggal dia ngetweetnya
        
## embeddings
this is a [pooled document embeddings](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/DOCUMENT_POOL_EMBEDDINGS.md) on: 
### flair
- [x] pretrain with lm-forward + tweets
- [ ] make tweet encoder
flair model can be downloaded [here](https://drive.google.com/drive/u/5/folders/1uLGvvNCNAjAeOBPKyMwtLfEErBAsYuMQ) 
### fasttext-id
- `WordEmbeddings('id-crawl')`

## search engine
[milvus](https://milvus.io/)
- make embedding:
	- [x] tfidf, done `tfidf.pkl`
	- [ ] fasttext
	- [x] flairembeddings
		- ~~ValueError: Found array with dim 3. check_pairwise_arrays expected <= 2. gak tau padahal gak adayang bikin dimensi 3~~ ganti ke scipy

id nya ikut di `0_koinworks_raw.csv` udah dibikin `uuid4` biar gampang bikin indexernya

## lda buat nyari complain
- dari lda didapet topic 23,44, 27
- topic 23:
	ini bentuknya ada investasi yang belum dikembalikan
- topic 44:
	ini bentuknya "jawaban" atau reply twitter dari koinworks, kadang ngereferensiin keluhan pelanggan
- topic 27:
	bentuknya komplain aplikasi, terkait, error, foto selfie, website yagn gak beres
setelah diambil dari tweet yang bukan dari koinworks didapat topic nomor: 14, 22, 34, 42
- di csv koinworks_keluhan_lda:
	- topic 12 itu kebanyakan promosi dan suruh cek dm
	- topic 11 is definitely keluhan

## blog post ideas
- [ini buat opening](https://twitter.com/pakelagu/status/1292346337803923456)
	- meme: top: SHARE KODE KW
	- meme: bottom: KU TERTYPU OLEH KW
### extras
- aplikasinya sempet ilang juga lol  cek id: 517, 529, , cek tanggal, cek sumber
- dari search twitter sempet peak di 263 tweet di 04-02-2020 dan 09-01-2020
- dari kmeans, langsung kepisah dengan cantik 3 label

- siap, [didanai](https://money.kompas.com/read/2020/05/18/130309726/koinworks-dapat-pendanaan-rp-149-miliar-dari-perusahaan-inggris?utm_source=dlvr.it&utm_medium=twitter) tapi tenor kagak dibayar :)))
	- [sumber2](https://medium.com/lendable/koinworks-secures-us-10-million-from-lendable-to-support-indonesias-digital-smes-7119f42f7809)
	- [sumber3](https://internationalfinance.com/koinworks-secures-10-mn-funding-help-smes-raise-funds-online/)

- didanai lagi dong [woah](pic.twitter.com/ZbFjMJ3aSp)
