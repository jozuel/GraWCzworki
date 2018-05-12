
import tensorflow as tf
import random

#wypełnia ilość elementów w tablicy do 25
def fillWithZero(lista):
    zero_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for _ in range(25-len(lista)):
        lista.append(zero_list.copy())

    return lista
#struktura danych pierwsze 7 znaków określa kolumne, kolejne 7 znaktów wiersz, kolejne 2 znaki gracza, ostatnie 3 znaki status rozgrywki(wygrana/przegrana/brak)
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
batch_size = 20000
display_step = 200

# Network Parameters
seq_max_len = 25 # Sequence max length                  #maxymalna długość 
n_hidden = 200 # hidden layer num of features           # iloś węzłów
n_classes = 7  # dane wyjściowe



# tf Graph input
x = tf.placeholder("float", [1,seq_max_len, 19])    #miejsce na dane typu float
y = tf.placeholder("float", [1,n_classes])          
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])   #miejsce na długość sekwencji

# Define weights        #ustawiamy losowe wartości dla wag i biases
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
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,sequence_length=seqlen)

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
    idx = tf.range(batch_size)*tf.shape(outputs)[1] + (seqlen - 1)  #last output index  #pobieramy ostatni output z ustalonej długości
    
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), idx)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']   #liczy wynik
def rnn_train():
    pred = dynamicRNN(x, seqlen, weights, biases)   #przewidujemy wynik

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    saver = tf.train.Saver()    
    with tf.Session() as sess:
        saver = tf.train.Saver()
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
        no_of_batches = 10   #ilosc paczek. ilosc danych / wielkosc paczki    
        epoch = 20 #ile razy ma sie cykl powtarzac
        for i in range(epoch):  
            ptr = 0
            for game in range(1000):      # do naszegame sieci neuronowej można by co 1 rozgrywkę optymalizować
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
                    sess.run(optimizer,feed_dict={x: [x_stack_list_copy], y: [y_train[game][j][0:7]],seqlen:[j+1]}) 
                    acc, loss = sess.run([accuracy, cost], feed_dict={x: [x_stack_list_copy], y: [y_train[game][j][0:7]], 
                                                    seqlen: [j+1]}) #0:7 by zabrać pierwsze 7 znaków określające kolumne
            for game in range(1000, 15000):      # do naszegame sieci neuronowej można by co 1 rozgrywkę optymalizować
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
                x_stack_list = x_train[game].copy()
                x_stack_list = fillWithZero(x_stack_list)
                sess.run(optimizer,feed_dict={x: [x_stack_list], y: [y_train[game][x_length-1][0:7]],seqlen:[x_length]}) 

            print("Step " + str(i) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

        print("Optimization Finished!")
        #save_path = saver.save(sess, "/tmp/model.ckpt")

        saver.save(sess, r'C:\Users\jozuel\Desktop\neural networks\my_test_model.ckpt')     #zapisujemy model

#rnn_train()
#init = tf.global_variables_initializer()
pred = dynamicRNN(x, seqlen, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, r'C:\Users\jozuel\Desktop\neural networks\my_test_model.ckpt')
    print("Model restored.")

   # sess.run(init)
    line = []
    lineList = []
    lineList_copy = []
    x_train = []
    y_train = []    #pobieranie danych i wynikow bez wypelnienia zerami
    x_stack_list = []
    x_stack_list_copy = []
    for i in range(25):
        line = input()
        line = line.split(",")  
        line = (list(map(int, line)))
        lineList.append(line.copy())
        lineList_copy = lineList.copy()
        lineList_copy = fillWithZero(lineList_copy)
        #print (sess.run(pred,feed_dict={x: [lineList_copy], seqlen:[i+1]}))
        output = sess.run(pred,feed_dict={x: [lineList_copy], seqlen:[i+1]})
        print(output)
        output = output[0]
        output = output[0:7]
        #print (sess.run(l1,{input_data: [line]}))
        maximum = output[0]
        index = 0
        for elem in range(len(output)):
            if([output[elem]]>maximum):
                maximum = output[elem]
                index = elem
        print(index)
