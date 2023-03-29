#######
# This code was written by Nino Verwei, Niek van Hilten, and Andrius Bernatavicius at Leiden University, The Netherlands (29 March 2023)
#
# When using this code, please cite:
# Van Hilten, N.; Verwei, N.; Methorst, J.; Nase, C.; Bernatavicius, A.; Risselada, H.J., biorxiv (2023)
#######


import tensorflow as tf
from tensorflow import keras as k
from keras import layers
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split, KFold
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib
tf.random.set_seed(373)
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # switch off GPU

shuffle = False

flags = argparse.ArgumentParser()

flags.add_argument("-f", "--input", default="../data/training_set.txt", 
            help="Input file with sequences in first column, scores in second column", type=str)
flags.add_argument("-v", "--val_data", default="../data/validation_set.txt", 
            help="Validation data with sequences in first column, scores in second column", type=str)
flags.add_argument("-a", "--alphabet", default="ARNDCFQEGHILKMPSTWYV", 
            help="Alphabet for one-hot encoding", type=str)
flags.add_argument("-r", "--lrate", default=0.0001,
            help="Learning rate", type=float)
flags.add_argument("-u", "--drate", default=0.052,
            help="Dropout rate", type=float)
flags.add_argument("-l", "--max_len", default= 24,
            help="max length of the sequnce.", type=int)
flags.add_argument("-e", "--embed_dim", default= 247,
            help = "embedding dimension for multihead attention and dense layer. Embedding dimension divided by number of head needs to result in an integer", type = int)
flags.add_argument("-z", "--num_heads", default= 8,
            help = "number of heads for multihead attention. Embedding dimension divided by number of head needs to result in an integer", type = int)
flags.add_argument("-k", "--n_transformblocks", default= 6,
            help = "amount of sequential transformerblocks", type = int)
flags.add_argument("-p", "--dense_dim", default= 38,
            help = "dimensions of last dense layer", type = int)
flags.add_argument("-i", "--epochs", default=75, 
            help="Initial number of epochs for training", type=int)
flags.add_argument("-b", "--batch", default=64,
            help="Batch size", type=int)

matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams['axes.axisbelow'] = True

args = flags.parse_args()


def get_data(filename): #obtain data and seperate sequences and fitness
    sequence = []
    fitness = []
    print(filename)
    n= 0
    with open(filename, "r") as dataset:
        for line in dataset:
            line = line.strip()
            if not line:
                continue

            line = line.split()
            sequence.append(line[0])
            fitness.append(line[1])
    
    sequences = np.array(sequence)
    seq_len = len(sequences[0])
    fitness_float =[]  #convert strings of fitness to floats in new list
    for index in range(len(fitness)):
        fitness_float.append(np.float32(fitness[index]))
    fitness = np.array(fitness_float)

    return(sequences, fitness, seq_len)

def get_data_random_seq(filename):
    sequence = []
    fitness = []
    with open(filename, "r") as dataset:
        n=0
        for line in dataset:
            if n == 0:
                n+= 1
                continue
            line = line.strip()

            if not line:
                continue

            line = line.split()
            sequence.append(line[0])
            fitness.append(line[1])
    sequences = np.array(sequence)
    seq_len = len(sequences[0])
    fitness_float =[]  #convert strings of fitness to floats in new list
    for index in range(len(fitness)):
        fitness_float.append(np.float32(fitness[index]))
    fitness = np.array(fitness_float)
    return(sequences, fitness, seq_len)

def preprocess_data(file_name, max_len):
    sequences, fitness, seq_len = get_data(file_name)
    Y = fitness
        
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(sequences)
    t_sequences = tokenizer.texts_to_sequences(sequences)

    t_p_sequences = k.preprocessing.sequence.pad_sequences(t_sequences, maxlen=max_len, padding='post')
    
    return t_p_sequences,fitness, tokenizer

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, drate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, kernel_initializer=k.initializers.glorot_uniform(seed=9456)) ####################
        self.ffn = k.Sequential(
            [layers.Dense(ff_dim, activation="relu", kernel_initializer=k.initializers.glorot_uniform(seed=535)), layers.Dense(embed_dim, kernel_initializer=k.initializers.glorot_uniform(seed=164)),] ####################
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(drate, seed = 867) ####################
        self.dropout2 = layers.Dropout(drate, seed = 236) ####################
    
    def get_config(self):
        config = super().get_config()
        config.update({"att" : self.att,"ffn" :self.ffn,"layernorm1" : self.layernorm1,"layernorm2" : self.layernorm2,"dropout1" : self.dropout1,"dropout2" : self.dropout2})
        return config

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, embeddings_initializer=k.initializers.glorot_uniform(seed=666)) ####################
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim, embeddings_initializer=k.initializers.glorot_uniform(seed=836)) ####################

    def get_config(self):
        config = super().get_config()
        config.update({"token_emb" : self.token_emb,"pos_emb" :self.pos_emb})
        return config

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def add_transformer_block(x, embed_dim, num_heads, ff_dim, drate):
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, drate)
    x = transformer_block(x)
    return x

def define_model(max_len = args.max_len, vocab_size = len(args.alphabet)+1, n_transformblocks = args.n_transformblocks, num_heads = args.num_heads, drate = args.drate, embed_dim =args.embed_dim, dense_dim = args.dense_dim):
    inputs = layers.Input(shape=(max_len,))
    embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, int(embed_dim))
    x = embedding_layer(inputs)
    for n in range(int(n_transformblocks)):
        x = add_transformer_block(x, embed_dim, num_heads, embed_dim, drate)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(drate, seed=34)(x)  ####################
    x = layers.Dense(dense_dim, activation="relu", kernel_initializer=k.initializers.glorot_uniform(seed=134))(x) ####################

    outputs = layers.Dense(1, activation=None, kernel_initializer=k.initializers.glorot_uniform(seed=147))(x) ####################

    model = k.Model(inputs=inputs, outputs=outputs)
    opt = k.optimizers.Adamax(learning_rate=args.lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    model.compile(optimizer=opt, loss="mean_squared_error")
    return model

def scheduler(epoch, lr):
    if epoch < 30:
        new_lr  = lr - (0.00009/30) # from 0.0001 to 0.00001 in 30 epochs
        return new_lr
    else:
        return lr       

class MyCustomCallback2(tf.keras.callbacks.Callback):
    def __init__(self, x, y,mse_res ):
        super(MyCustomCallback2, self).__init__()
        self.x = x
        self.mse_res = mse_res
        self.y = y

    def on_epoch_end(self, epoch, logs=None):
        res_eval_1 = self.model.evaluate(self.x, self.y, verbose = 0)
        self.mse_res.append(res_eval_1)
        print(res_eval_1)  

#############################################################################################################################



vocab_size = len(args.alphabet)+1 # alphabet and 0 token

t_p_sequences,fitness, tokenizer = preprocess_data(args.input, args.max_len)

tf.keras.backend.clear_session()

r_sequences, r_fitness, seq_len = get_data_random_seq(args.val_data)

r_t_sequences = tokenizer.texts_to_sequences(r_sequences)
r_pt_sequences = k.preprocessing.sequence.pad_sequences(r_t_sequences, maxlen=args.max_len, padding='post')

mse_res = []
n= 0
model = define_model(max_len = args.max_len, vocab_size = len(args.alphabet)+1, n_transformblocks = 6, num_heads = 8, drate = 0.052348568295771794, embed_dim =247, dense_dim = 38)
print(model.summary())
es = k.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
lr = k.callbacks.LearningRateScheduler(scheduler)
his2 = MyCustomCallback2(r_pt_sequences, r_fitness, mse_res)
history = model.fit(
                    t_p_sequences,fitness, 
                    epochs= args.epochs, 
                    batch_size=args.batch,
                    validation_data=(r_pt_sequences, r_fitness),
                    verbose=1,
                    shuffle = shuffle,
                    callbacks=[es, his2, lr])
print(mse_res)
                

#plot loss
plt.plot(history.history['loss'], color='blue', label='Training')
plt.plot(history.history['val_loss'], color='green', label='Validation')
plt.grid(alpha=0.5)
plt.xlabel("Epoch")
plt.ylabel("Mean squared error (MSE) (kJ$^{2}$ mol$^{-2}$)")
plt.legend(['Training', 'Validation'])
plt.show()
plt.savefig("training.pdf")
plt.close()

print(history.history['loss'])
print("final min loss:", min(history.history['loss']))
print(history.history['val_loss'])
print("final min vall loss:",min(history.history['val_loss']))


with open("losses.dat", "wb") as f:
    pickle.dump([history.history['loss'],history.history['val_loss']],f)

