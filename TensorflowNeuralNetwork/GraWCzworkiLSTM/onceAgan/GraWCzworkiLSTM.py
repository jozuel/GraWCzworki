import tensorflow as tf

def openFile():
    file = open("test.txt", "r")
    lista = []
    x = []
    list_of_x = []
    list_of_y = []
    y = []
    tmp = []
    #zero_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
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
       # for _ in range(25-len(x)):
       #     x.append(zero_list.copy())
       #     y.append(zero_list.copy())
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

#dane z pliku by trzeba podzielic na train input i train output były by podobne tylko przesuniete o 1 dalej
input_data = tf.placeholder(tf.float32,[1,19]) #20 bitow po 1 bicie na raz # u nas by było chyba None,19
target = tf.placeholder(tf.float32,[1,19])  # poprawny output one_hot od 0 do 20    # u nas by było 19 bo tak samo jak input

num_hidden = 24
cell = tf.nn.rnn_cell.LSTMCell(num_hidden) #num_hidden to ilosc ukrytych wezlow

val, state = tf.nn.static_rnn(cell,[input_data], dtype=tf.float32)   #po uzyciu tego przepuszczamy input_data przez siec. 
#val to output dla każdego etapu state to status ale nie będzie on wykozystywany
#val = tf.transpose(val,[1,0,2])     #zmieniamy tak by paczka była sekwencją co kolwiek to znaczy. Chyba to jest tak ze jak np mamy paczke 20 elementow to twozy 20 X 1 elementow
#last = tf.gather(val, int(val.get_shape()[0])-1)    #zbieramy ostatni output

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))     #number of hidden x number of classes
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))                     #batch size x number of classes

l1 = tf.add(tf.matmul(val[-1],weight),bias)
l1 = tf.nn.relu(l1)


#prediction = tf.nn.softmax(tf.matmul(val, weight) + bias) #softmax oblicza prawdopodobienstwo wyniku czy cos w tym stylu i zwraca wektor
cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=l1, labels=input_data) )
#cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))     #obliczamy jak blisko bylismy wyniku i bezemy z tego logarytm

optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)    #optymalizujemy błąd. Jakoś to się robi

mistakes = tf.not_equal(tf.arg_max(target,1),tf.arg_max(l1,1)) #sprawdzamy czy mamy blad. 2 parametr arg_maxa to rozmiar
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))   #liczymy ile danych obliczylismy zle

cell = cell.zero_state(1,tf.float32)
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    x = []
    y = []

    x, y = openFile()
    #x = slice_it(x,19)
    #y = slice_it(y,19)
    sess.run(init_op)   #uruchamiamy sesje i inicjalizujemy zmienne (te na samej goze)
    batch_size = 1000   #wielkosc paczki
    no_of_batches = 100    #ilosc paczek. ilosc danych / wielkosc paczki    
    epoch = 5 #ile razy ma sie cykl powtarzac
    for i in range(epoch):  
        ptr = 0
        for game in range(no_of_batches):      # do naszegame sieci neuronowej można by co 1 rozgrywkę optymalizować
            #inp, out = train_input[ptr:ptr+ batch_size], train_output[ptr:ptr+batch_size]   #pobieramy wejscie i wyjscie od ptr do ptr + wielkosc paczki
            ptr+=batch_size;
            if(len(x[game])>len(y[game])):
                minimum = len(x[game])
            else:
                minimum = len(y[game])
            minimum -= 1
            for round in range(minimum):
                x_round_list = []
                y_round_list = []
                x_round_list.append(x[game][round])
                y_round_list.append(y[game][round])
                sess.run(minimize,{input_data: x_round_list, target: y_round_list})   #obliczamy błąd bo siec liczy z inputu i dostaje poprawny output(target).
                x_round_list.pop()
                y_round_list.pop()
            sess.run(cell)
        print("Epoch ",str(i))
    for game in range(no_of_batches+1000):
        if(len(x[game])>len(y[game])):
            minimum = len(x[game])
        else:
            minimum = len(y[game])
        minimum -= 1
        for round in range(minimum):
            x_round_list.append(x[game][round])
            y_round_list.append(y[game][round])
            incorrect = sess.run(error, {input_data: x_round_list, target: y_round_list})  #sprawdzamy jak bardzo sie myli po wytrenowaniu
            x_round_list.pop()
            y_round_list.pop()
    print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))

    print (sess.run(l1,{input_data: [[0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0]]}))    #sprawdzenie jak dziala