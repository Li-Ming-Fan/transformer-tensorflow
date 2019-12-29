# Transformer-Tensorflow

Transformer, Tensorflow. Dealing with a toy task: to copy a sequence of numbers.

### Example results

Eval results after training 9000 batches:

<img src="https://github.com/Li-Ming-Fan/transformer-tensorflow/blob/master/aaa_task_copy_result_examples/eval_result.PNG" width="384">


### Description
  
To run this repo:

```
python task_copy_data_set.py            # 1, to create vocab
python script_runner.py --mode=train    # 2, to train and validate
python script_runner.py --mode=eval     # 3, to evaluate
```
  
</br>

by 1, directory ./vocab/ and file ./vocab/vocab_tokens.txt will be created. 
  
by 2, directory ./task_copy_results/ and 3 subdirectories will be created, and training log and model ckpt will be stored in subdirectories.
  
by 3, the model will be run through an evaluation.


</br>

Tested with tensorflow version 1.8.0

Using Zeras for model baseboard: pip install Zeras==0.4.3


### Reference
  
1, Attention is All You Need,

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, https://arxiv.org/abs/1706.03762

2, The Annotated Transformer, http://nlp.seas.harvard.edu/2018/04/03/attention.html



