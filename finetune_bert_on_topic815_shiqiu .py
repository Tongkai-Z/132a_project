# -*- coding: utf-8 -*-
"""Finetune_Bert_on_topic815_ShiQiu.ipynb

Author: Shi Qiu
This is the python script form of our Jupyter Notebook for fine tune bert.
Please note:
Due to the computational power limitation, we have to use Google Colab to train the bert model. So this script is not runnable locally.
Please refer to the link below to see a runnable version of this code in Jupyter notebook.

Original file is located at
    https://colab.research.google.com/drive/1idHtwMeycLTqmv7GGgyvvFmk6TOp8O6l

# Fine Tune Bert on Topic815

## Setup GPU
We first set up the GPU's provided in Google Colab. Due to the computational power limitation, we can hardly train the bert model on our local machines. 

In this google colab environment, we can easily access some GPUs for our training code.
"""

import tensorflow as tf

# Get the GPU device name.
# Print message if connect to GPU successfully.
device_name = tf.test.gpu_device_name()

if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')

import torch
# Setup GPU for pytorch
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('Use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

"""## Setup Hugging Face
Hugging face library seems to be an well-known library working with Bert. We will be using hugging face library with pytorch to fine tune our Bert model. Also our training will be built-on some pre-trained model from hugging face library.
"""

# install transformer package from hugging face, which is a pytorch interface
!pip install transformers

"""## Load Training data
We extracted all relevant documents locally from provided TREC corpus, and formated them into proper csv format. 

We uploaded the processed csv file into google drive, and access the external training data from google drive by `google.colab drive` module. Please refer to [here](https://colab.research.google.com/notebooks/io.ipynb#scrollTo=RWSJpsyKqHjH) for loading file from your google drive.
"""

from google.colab import drive
drive.mount('/content/drive')

# Locate the file in my google drive
!ls '/content/drive/MyDrive/Colab Notebooks'

import pandas as pd

# Load training data to dataframe, print any 5 random samples.
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/815_relevant_docs.csv', delimiter=',', header=1, names=['label', 'content', 'customed_content'])
print(df.shape[0])

# Manually detected row 303 contains Nan, drop that row.
df.dropna(subset = ["content"], inplace=True)
print(df.shape[0])

"""We will use the `label` as the ground truth in training, and customed_content as training data.
Note: We simplified the task into a 2-classification job. Any document with label 1 is the relevant document, while document with label 0 are irrelevant.
"""

# Get the lists of sentences and their labels.
contents = df.customed_content.values
labels = df.label.values
contents[0]

"""## Tokenization & Input Formatting
After getting the raw input content, we will look into transforming the raw content into the format which will be used to fine-tune our bert model.

1. Tokenize the content string with bert tokenizer
2. Convert the into a vector of integer for later training purpose.
"""

# Tokenize the raw content
from transformers import BertTokenizer

# Load the BERT tokenizer, using the tokenizer from 'bert-base-uncased' model
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-uncased', do_lower_case=True)

"""Look at one sample document, and view the tokenized vector."""

print('Tokenized: ', tokenizer.tokenize(contents[0]))

print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(contents[0])))

"""**Problem:** The current vector only has max length of 512, but the actual size of a document is far longer than this."""

# # Calculate maximum document length in corpus. Create tokenized vector for training
# max_len = 0
# # For every document
# # TODO: Find a way to fix max_length problem
# for doc in contents:

#     # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
#     input_ids = tokenizer.encode(doc, add_special_tokens=True, max_length=512)

#     # Update the maximum sentence length.
#     max_len = max(max_len, len(input_ids))

# print('Max sentence length: ', max_len)

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for doc in contents:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        doc,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 128,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Print sentence 0, now as a list of IDs.
print('Original: ', contents[0])
print('Token IDs:', input_ids[0])

"""## Split Training Data and Test data

Convert the content vector to Tensor, and split into ratio of 9:1. 90% are used as training data, 10% used for validation data.
"""

from torch.utils.data import TensorDataset, random_split

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# Calculate the number of samples to include in each set.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

"""## Setup model and training
We will specify the detailed config of our model in this section. 

In summary, we used the distiled bert model as is HW5. The hyperparameter settings are as following:
    * batch size 8
    * Epoch number 4
    * Learning rate 0.05
    * Adam optimizer
Due to the memory limitation in Colab, we cannot use any larger batch size. 

The origin bert model has 12 layers, with 786 feature size in each layer.

We also convert the given distilled bert model into a binary classification model, where the 1's are relevant document, and the 0 are irrelevant documents. We are not exactly sure if this approach is the best way to fine tune bert. But ideally, we hope that the model will learn to classify between relevant and ireelevant documents, and representing better results in embeddings.
"""

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# we specify the batch size here. Due to the memory issue, we cannot use any larger
# batch size, but have to fix it at size 8.
batch_size = 8

# Create the DataLoaders for our training and validation sets.
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

from transformers import BertForSequenceClassification, AdamW, BertConfig

# Load BertForSequenceClassification using distilled bert as baseline,
# the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "sentence-transformers/msmarco-distilbert-base-v3", # Use the distil-berted in PA5
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # Convert this model into a classification model 
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()

# Specify the optimizer and learning rate
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate
                  eps = 1e-8 # args.adam_epsilon 
                )

import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

from transformers import get_linear_schedule_with_warmup

# Number of training epochs. The BERT authors recommend between 2 and 4. 
# We chose to run for 4 for this experiment.
epochs = 4

# Total number of training steps is [number of batches] x [number of epochs]. 
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

import random
import numpy as np
import time
import datetime

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()
torch.cuda.empty_cache()


for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Clear gradient descent
        model.zero_grad()        

        # Perform a forward pass
        result = model(b_input_ids, 
                       token_type_ids=None, 
                       attention_mask=b_input_mask, 
                       labels=b_labels,
                       return_dict=True)

        loss = result.loss
        logits = result.logits

        # Accumulate the training loss over
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # Measure how long this epoch took.
    training_time = time.time() - t0

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Unpack this training batch from our dataloader. 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            result = model(b_input_ids, 
                           token_type_ids=None, 
                           attention_mask=b_input_mask,
                           labels=b_labels,
                           return_dict=True)

        # Get the loss and "logits" output by the model. The "logits" are the 
        # output values prior to applying an activation function like the 
        # softmax.
        loss = result.loss
        logits = result.logits
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = time.time() - t0
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(time.time()-total_t0))

"""## Save Model

After finish training, we will now save the model to google drive for later use in our elastic search. The saved model can be found [here.](https://drive.google.com/drive/folders/1jBJW8qYPE87ELZzB7Rz1nNbjPA3qkPlJ?usp=sharing)
"""

import os

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

output_dir = './model_save/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))

!cp -r ./model_save/ "./drive/MyDrive/Colab Notebooks"