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
- [x] drop duplicate tweets: 
	- promo2 gak jelas -> biasanya bot
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
id: [11, 262, 282, 368, 379, 398, 404, 441, 516, 530, 555, 642, 660, 702, 799, 806, 819, 872, 1241, 1242, 1243, 1415,1609 ]

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
