# Observations

1. Learning Rate 
    1. 0.001 is too small -> 0.7597 mic-avg

2. Results:

| Hidden        | Embeddings		| Lineup 	| lr 	|dropout |F1-micro|Accuracy|
| ------------- |:-------------:	| -------------:|-------|--------|--------|--------| 
| 512      	| elmo-original-all  	| biLSTM 	|0.1 	|   0	 |81.9    |90      |
| 512		| elmo-original-all    	| biLSTM+CRF 	|0.1 	|   0	 |82.63   |90.4    |  
| 512		| elmo-oa + vembed	| biLSTM+CRF	|0.001	|   0	 |75.56	  |77.34   |
| 2048		| elmo-oa + vembed 	| biLSTM+CRF	|0.2	|  0.1   |81.72   |83.21   |  
| 2048		| elmo-oa + vembed 	| biLSTM+CRF	|0.2	|  0.2   |80.95   |82.53   |
| 1024		| elmo-oa + vembed 	| biLSTM+CRF	|0.2	|  0.2   |82.06   |83.61   |
| 300		| elmo-oa + vembed	| biLSTM+CRF	|0.2	|  0.4	 |74.08	  |76.05   |
| 512		| elmo-sav + pos	| biLSTM	|0.02	|  0.3	 |79.83	  |81.16   |
| 256		| elmo-sav + pos ->256	| biLSTM	|0.02	|  0.3	 |78.50	  |80.09   |


3. Questions for Akbik:
	+ Für Verben ist es immer noch möglich eine weitere Schicht einzubeziehen indem man gar nicht verschlüsselt dass ein Verb tatsächlich ein Verb ist und stattdessen eine Role wenn es überhaupt eine andere spielt. Dass es ein verb ist kann man mit einem Embedding weitergeben. Das Problem ist jedoch dass ich es nicht schaffen konnte :)
