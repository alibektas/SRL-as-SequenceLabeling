# Observations

1. Learning Rate 
    1. 0.001 is too small -> 0.7597 mic-avg

2. Results:

| Hidden        | Embeddings		| Lineup 	    | lr 	|dropout |F1-micro|Accuracy | Info  |
| ------------- |:-------------:	| -------------:|-------|--------|--------|---------| ----- |
| 512      	| elmo-original-all  	| biLSTM 	    |0.1    |   0	 |81.9    |90       |       |
| 512      	| elmo-original-all  	| biLSTM 	    |0.5    |   0.05 |82.36   |83.44    |       |
| 512       | elmo-oa               | biLSTM        |0.02   |   0.3  |79.7    |81.17    |       |
| 512       | elmo-oa               | biLSTM        |0.02   |   0.05 |80.11   |81.55    |       |
| 512		| elmo-original-all    	| biLSTM+CRF    |0.1 	|   0	 |82.63   |90.4     |       |
| 512		| elmo-oa + vembed	    | biLSTM+CRF	|0.001	|   0	 |75.56	  |77.34    |       |
| 2048		| elmo-oa + vembed 	    | biLSTM+CRF	|0.2	|  0.1   |81.72   |83.21    |       |
| 2048		| elmo-oa + vembed 	    | biLSTM+CRF	|0.2	|  0.2   |80.95   |82.53    |       |
| 1024		| elmo-oa + vembed 	    | biLSTM+CRF	|0.2	|  0.2   |82.06   |83.61    |       |
| 300		| elmo-oa + vembed	    | biLSTM+CRF	|0.2	|  0.4	 |74.08	  |76.05    |       |
| 512		| elmo-sav + pos	    | biLSTM	    |0.02	|  0.3	 |79.83	  |81.16    |                     |
| 256		| elmo-sav + pos ->256	| biLSTM	    |0.02	|  0.3	 |78.50	  |80.09    |                     |
| 512       | elmo-oa               | biLSTM        |0.1    |   0    |97.28   |98.96    | Predicate Prediction|
| 1024      | elmo-oa + vembed      | biLSTM        | 0.1   |   0    |87.43   |87.95    | Direction Prediction|
| 512       | elmo-oa               | biLSTM        | 0.02  |   0    |79.86   |81.28    | +top-4 RolePairs    |


1. How to resolve this? Token indices sequence length is longer than the specified maximum sequence length for this model (668 > 512). Running this sequence through the model will result in indexing errors
2. In dataset_dev.entries[13].get_span() this example there is a duplicate of the same ARGM role. I should check if it is allowed to use duplicates. ( I know that it isn't for the common roles ARGX , but not quite sure if the same rule applies  for modifiers.)

Belki V>> gibi tagler ekleyerek verbleri birbirine baglayabilirim.
Neden bunu yaptim , onemli. 
Herhangi bir ornekte bunu gormek kolay olmasa da verb olarak isaretlenen bircok sey aslinda basit rollerde bulunan kelimeler.
Give me a copy of this cumlesinde copy verbden cok give in ARG2'si olarak gorulmelidir. En azindan copy'nin baska bi fiile bagli oldugunu soylemek onun icin bir rolu oldugunu gosterir. Buna ek olarak aslinda su dusunulebilir: "Verbler yerine rolleri one cikar. Verbleri zaten ayri tahmin eden bir yapin olacak" 

Yakin roleu uzak olana tercih et.

yarina spanisation kaldi. experimentler yapiliyor , sonra bu experimentlerden bir de bert icin yapilacak.



## TODOs 
1. ID:82 How come this entry has "to be" as its verb yet all roles are connected to the noun "weapons"?



## Akbike sorular
1. Belki henuz o noktada olmasam da biraz daha ilerledikten sonra , heuristics olmadan sonuclari kabul edilebilir bir noktaya getirecmeyecegim. Tabiii ki bu noktaya gelmeden once 
nasil labels sayisinin bir role word e birden fazla rol eklemeyle exponensiyel arttigini gosterecegim. Peki ya sonrasi? Heuristics yani ne kadar isteniyor , bekleniyor?