""" Dynamic Recurrent Neural Network.
TensorFlow implementation of a Recurrent Neural Network (LSTM) that performs
dynamic computation over sequences with variable length. This example is using
a toy dataset to classify linear sequences. The generated sequences have
variable length.
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
import random


# ====================
#  DATA READER
# ====================
class ToySequenceData(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, n_samples=1000, max_seq_len=25, min_seq_len=1,
                 max_value=1000):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            # Random sequence length
            len = random.randint(min_seq_len, max_seq_len)
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(len)
            # Add a random or linear int sequence (50% prob)
            if random.random() < .5:
                # Generate a linear sequence
                rand_start = random.randint(0, max_value - len)
                s = [[float(i)/max_value] for i in
                     range(rand_start, rand_start + len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([1., 0.])
            else:
                # Generate a random sequence
                s = [[float(random.randint(0, max_value))/max_value]
                     for i in range(len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([0., 1.])
        self.batch_id = 0
    
    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        # tu można podstawić funkcję z pobieraniem z pliku
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen
def fillWithZero(lista):
    zero_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for _ in range(25-len(lista)):
        lista.append(zero_list.copy())

    return lista
def openFile():
    file = open("test.txt", "r")
    lista = []
    x = []
    list_of_x = []
    list_of_y = []
    y = []
    tmp = []
    zero_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    lines = file.readlines()    # czyta wszystkie linie
    for line in lines:      
        line = line.replace(" ", "")
        line = line.split(",")  
        line.pop()      #usuwa ostani element
        lista.append(list(map(int, line)))  #mapuje stringa na inta

    for i in range(len(lista)):
        tmp = slice_it(lista[i],19)
        for j in range(len(tmp)):
            if(j % 2==0):
                x.append(tmp[j])
            else:
                y.append(tmp[j])
       # x = fillWithZero(x)    #wypelniac zerem bedziemy po pobraniu danych by miec info o ilosci elementow
       # y = fillWithZero(y)
        #for _ in range(25-len(x)):
        #    x.append(zero_list.copy())
        #    y.append(zero_list.copy())
        list_of_x.append(x.copy())
        list_of_y.append(y.copy())
        x.clear()
        y.clear()

        #if i % 2==0:    #bierzemy co 2 by na zmiane był gracz i komputer obliczał gdzie ma się ruszyć potem sprawdzamy z Y
        #   x.append(lista[i])
        #else:
        #    y.append(lista[i])

    #lista = list(map(int, lista))   #zmiana liter na cyfry
    return list_of_x, list_of_y    #lista to jest ciąg bitów dla całej sekwencji. Trzeba podzielić przez 19 by mieć pojedyńczy element sekwencji

def slice_it(arr, size):
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr = arr[size:]
    arrs.append(arr)
    return arrs


# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_steps = 10000
batch_size = 10000
display_step = 200

# Network Parameters
seq_max_len = 25 # Sequence max length
n_hidden = 64 # hidden layer num of features
n_classes = 19 # linear sequence or not

#trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
#testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)

# tf Graph input
x = tf.placeholder("float", [1,seq_max_len, 19])
y = tf.placeholder("float", [1,n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def dynamicRNN(x, seqlen, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,state_is_tuple=True)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    batch_size = tf.shape(outputs)[0]
    # Hack to build the indexing and retrieve the right output.
    idx = tf.range(batch_size)*tf.shape(outputs)[1] + (seqlen - 1)  #last output index
    
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), idx)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

pred = dynamicRNN(x, seqlen, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    lineList = []
    lineList_copy = []
    x_train = []
    y_train = []    #pobieranie danych i wynikow bez wypelnienia zerami
    x_stack_list = []
    x_stack_list_copy = []
    x_train, y_train = openFile()
    #batch_size = 10000   #wielkosc paczki
    no_of_batches = 5000   #ilosc paczek. ilosc danych / wielkosc paczki    
    epoch = 5 #ile razy ma sie cykl powtarzac
    for i in range(epoch):  
        ptr = 0
        for game in range(batch_size):      # do naszegame sieci neuronowej można by co 1 rozgrywkę optymalizować
            #inp, out = train_input[ptr:ptr+ batch_size], train_output[ptr:ptr+batch_size]   #pobieramy wejscie i wyjscie od ptr do ptr + wielkosc paczki
            ptr+=batch_size;
            if(len(x_train[game])>len(y_train[game])):
                minimum = len(y_train[game])
                x_train[game].pop()
            elif(len(x_train[game])<len(y_train[game])):
                minimum = len(x_train[game])
                y_train[game].pop()             # w przypadku gdy 1 inputu jest za dużo (x lub y) by była ich rowna ilosc
            else:
                minimum = len(x_train[game])

            x_length = len(x_train[game])
            x_stack_list_copy.clear()
            x_stack_list.clear()
            for j in range(x_length):
                x_stack_list.append(x_train[game][j].copy())    
                x_stack_list_copy = x_stack_list.copy()
                x_stack_list_copy = fillWithZero(x_stack_list_copy)     #shape (25,19)
                sess.run(optimizer,feed_dict={x: [x_stack_list_copy], y: [y_train[game][j]],seqlen:[j+1]}) 
                acc, loss = sess.run([accuracy, cost], feed_dict={x: [x_stack_list_copy], y: [y_train[game][j]],
                                                seqlen: [j+1]})
                #dodac petle z onceagane by miec rosnaca tablice elementow x i 1 element y


            #for round in range(minimum):
             #   x_round_list = []
              #  y_round_list = []
               # x_round_list.append(x[game][round])
                #y_round_list.append(y[game][round])
            #sess.run(optimizer,feed_dict={x: [x_train[game]], y: [y_train[game]],seqlen:[minimum]}) 
        #acc, loss = sess.run([accuracy, cost], feed_dict={x: [x_train[game]], y: [y_train[game]],
        #                                        seqlen: minimum})
        print("Step " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
    #for step in range(1, training_steps + 1):
       # batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        # Run optimization op (backprop)
        #sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
        #                               seqlen: batch_seqlen})
        #if step % display_step == 0 or step == 1:
            # Calculate batch accuracy & loss
        #    acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y,
        #                                        seqlen: batch_seqlen})
        #    print("Step " + str(step*batch_size) + ", Minibatch Loss= " + \
        #          "{:.6f}".format(loss) + ", Training Accuracy= " + \
        #          "{:.5f}".format(acc))

    print("Optimization Finished!")
    for i in range(25):
        line = input()
        line = line.split(",")  
        line = (list(map(int, line)))
        lineList.append(line.copy())
        lineList_copy = lineList.copy()
        lineList_copy = fillWithZero(lineList_copy)
        print (sess.run(l1,{x: [lineList_copy], seqlen:[i+1]}))
    # Calculate accuracy
    #test_data = testset.data
    #test_label = testset.labels
    #test_seqlen = testset.seqlen
    #print("Testing Accuracy:", \
    #    sess.run(accuracy, feed_dict={x: test_data, y: test_label,
#seqlen: test_seqlen}))