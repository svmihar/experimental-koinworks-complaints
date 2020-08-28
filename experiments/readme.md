# top2vec

## troubles



### solved
```
keywords learned are only: 110

while all unique keywords are 5351

and most common words are:
[('to', 367),
 ('ya', 272),
 ('…', 266),
 ('bisa', 258),
 ('and', 243),
 ('ada', 236),
 ('indonesia', 224),
 ('financial', 196),
 ('get', 168),
 ('sudah', 162)]
 ```
 decreasing min_count, using ns_exponent to normalize high and low frequency words, decreasing window -> mainly cause tweets are <15 chars so why bother.