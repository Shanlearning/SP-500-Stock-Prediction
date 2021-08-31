import sys, os
from pathlib import Path
project_dir = Path.cwd()
os.chdir(project_dir)
sys.path.append(os.getcwd())
#os.chdir('change to the mother working directory')

###############################################################################
from pathlib import Path

import shutil
import logging
import torch

from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from sklearn.model_selection import train_test_split
from function.bert_setting import Config,customBert
from sklearn.metrics import classification_report
from torch.nn import CrossEntropyLoss
import pandas as pd
import numpy as np
###############################################################################

project_dir = Path.cwd()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

###############################################################################
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.ERROR)

###############################################################################

lm_path = project_dir/'model'/'language_model'
cl_path = project_dir/'model'/'output_model'/'stock'
cl_data_path = project_dir/'model'/'data'/'FinancialPhraseBank'

###############################################################################
# Clean the cl_path
try:
    shutil.rmtree(cl_path) 
except:
    pass
###############################################################################
# Get the train test examples
train_test = pd.read_csv('./model/data/FinancialPhraseBank/Sentences_66Agree.txt', header = None, sep='@', encoding='latin-1')
train_test.columns = ['text','label'] 

train, val = train_test_split(train_test, test_size=0.2)
train.to_csv("./model/data/FinancialPhraseBank/train.csv",encoding = "utf-8")
val.to_csv("./model/data/FinancialPhraseBank/validation.csv",encoding = "utf-8")
###############################################################################
label_list=['positive','negative','neutral']
bertmodel = BertForSequenceClassification.from_pretrained(lm_path,cache_dir=None, num_labels= len(label_list) )

config = Config(   data_dir=cl_data_path,
                   bert_model=bertmodel,
                   num_train_epochs=4,
                   model_dir=cl_path,
                   max_seq_length = 128,
                   train_batch_size = 16,
                   learning_rate = 2e-5,
                   output_mode='classification',
                   warm_up_proportion=0.2,
                   local_rank=-1,
                   discriminate=True,
                   gradual_unfreeze=True )
##############################################################################

customBert = customBert(config)
customBert.prepare_model(label_list)

###############################################################################

train_data = customBert.get_data('train')
model = customBert.create_the_model()

###############################################################################
# This is for fine-tuning a subset of the model.
freeze = 11

for param in model.bert.embeddings.parameters():
    param.requires_grad = False
    
for i in range(freeze):
    for param in model.bert.encoder.layer[i].parameters():
        param.requires_grad = False

###############################################################################

trained_model = customBert.train(train_examples = train_data, model = model)

###############################################################################

test_data = customBert.get_data('validation')
results = customBert.evaluate(examples=test_data, model=trained_model)

###############################################################################

results['prediction'] = results.predictions.apply(lambda x: np.argmax(x,axis=0))
print(np.mean(results['labels'] == results['prediction']))
print(np.mean(results['labels'] == results['prediction']))

np.sum(train['label'] == "positive")
np.sum(train['label'] == "negative")
np.sum(train['label'] == "neutral")

['positive','negative','neutral']
np.sum((results['labels'] ==2)&( results['prediction'] == 1))

def report(df, cols=['label','prediction','logits']):
    #print('Validation loss:{0:.2f}'.format(metrics['best_validation_loss']))
    cs = CrossEntropyLoss(weight=customBert.class_weights)
    loss = cs(torch.tensor(list(df[cols[2]])),torch.tensor(list(df[cols[0]])))
    print("Loss:{0:.2f}".format(loss))
    print("Accuracy:{0:.2f}".format((df[cols[0]] == df[cols[1]]).sum() / df.shape[0]) )
    print("\nClassification Report:")
    print(classification_report(df[cols[0]], df[cols[1]]))

report(results,cols=['labels','prediction','predictions'])




