## Decision Tree

Notes:
* Assume input files has no missing/null values

File structure:
* ID3.py: class DecisionTree, main_program
* preprocess.py: pre process function

Implement:
* decision tree vanilla
* decision tree with pruning (Reduce Error Pruning)
* decision tree with maxDepth

Commands
```
python3 ID3.py train-file test-file vanilla 80
python3 ID3.py train-file test-file prune 50 40
python3 ID3.py train-file test-file maxDepth 50 40 5
```

