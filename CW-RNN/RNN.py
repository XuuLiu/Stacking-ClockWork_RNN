import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_generator import *
import math

(X_train, y_train), (X_validation, y_validation) = generate_data("/Users/XuLiu/Documents/cwrnn/price_avg.txt")
ntrain,ntest=X_train.shape[0],X_validation.shape[0]


diminput=90 #lag
dimhidden=140 #
dimoutput=1 #one step ahead predict
nsteps=1
weights={
    'hidden':tf.Variable(tf.random_normal([diminput,dimhidden])),
    'out':tf.Variable(tf.random_normal([dimhidden,dimoutput]))
}
biases={
    'hidden':tf.Variable(tf.random_normal([dimhidden])),
    'out':tf.Variable(tf.random_normal([dimoutput]))
}

def _RNN(_X,_W,_b,_nsteps,_name):
    # 1. permute input from [batch_size,nstep,diminput] to [nsteps,batch_size,diminput]
    #_X=tf.transpose(_X,[1,0,2])
    # 2. reshape input to [nsteps*batch_size,diminput]
    _X=tf.reshape(_X,[-1,diminput])
    # 3.Input layer => hidden layer
    _H=tf.matmul(_X,_W['hidden'])+_b['hidden']
    # 4.splite data to 'nsteps' chunks, an i-th chunck indicateds i-th batch data
    _Hsplit=tf.split(_H,_nsteps,0)
    # 5. get LSTM's final output(_LSTM_O) and state(_LSTM_S)
    # Both _LSTM_O and _LSTM_S consist of 'batch_size' elements
    # only _LSTM_O with be used to predict the output
    with tf.variable_scope(_name) as scope:
        #scope.reuse_variables()
        lstm_cell=tf.contrib.rnn.BasicRNNCell(dimhidden) # activation default tanh
        _LSTM_O,_LSTM_S=tf.contrib.rnn.static_rnn(cell=lstm_cell,inputs=_Hsplit,dtype=tf.float32)
    # 6. output
    _O=tf.matmul(_LSTM_O[-1],_W['out'])+_b['out']
    # return
    return {
        'X':_X,
        'H':_H,
        'Hsplit':_Hsplit,
        'LSTM_O':_LSTM_O,
        'LSTM_S':_LSTM_S,
        'O':_O
    }

global_step=tf.Variable(0.001,name='global_step')
learning_rate = tf.train.exponential_decay(
    0.0001,
    global_step,
    1000,
    0.57,
    staircase=True
)
learning_rate = tf.maximum(learning_rate, 0.000001)


x=tf.placeholder('float',[None,diminput,nsteps])
y=tf.placeholder('float',[None,dimoutput])
myrnn=_RNN(x,weights,biases,nsteps,'basic')
pred=myrnn['O']
#error = tf.reduce_sum(tf.square(y - pred),axis=1)
#loss = tf.reduce_mean(error, name="loss")
loss=tf.reduce_mean(tf.square(tf.transpose(pred) - y))
optm=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#accr=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(y,1)),tf.float32))
init=tf.global_variables_initializer()

training_epoche=200
batch_size=55
display_step=1
sess=tf.Session()
sess.run(init)

for epoch in range(training_epoche):
    avg_loss=0.
    avg_loss_test=0.
    total_batch=int(ntrain/batch_size)
    pred_train=[]
    pred_test = []
    for i in range(total_batch):
        global_step=(epoch*i)+i+1
        index_start = i * batch_size
        index_end = index_start + batch_size
        batch_xs=X_train[index_start:index_end,]
        batch_ys=y_train[index_start:index_end,]
        batch_xs=batch_xs.reshape((batch_size,diminput,nsteps))
        feeds={x:batch_xs,y:batch_ys}
        _,w_value,predictions_train=sess.run([optm,weights,pred],feed_dict=feeds)
        pred_train.append(predictions_train)
        avg_loss+=sess.run(loss,feed_dict=feeds)/total_batch
    np.savetxt('./pred_train_rnn/pred_train_epoch%3d'%epoch,pred_train)

    if epoch%display_step==0:
        avg_loss_test=0
        print "Epoch:%03d/%03d cost: %.9f"%(epoch,training_epoche,avg_loss)
        #train_acc=sess.run(accr,feed_dict=feeds)
        #print "Training accuracy:%.3f"%(train_acc)
    for i in range(int(math.floor(ntest/batch_size))):
        index_start = i * batch_size
        index_end = index_start + batch_size
        batch_xt = X_validation[index_start:index_end, ]
        batch_yt = y_validation[index_start:index_end, ]
        batch_xt=batch_xt.reshape((batch_size,diminput,nsteps))
        feeds={x:batch_xt,y:batch_yt}
        avg_loss_test += sess.run(loss, feed_dict=feeds) / total_batch
        predictions=sess.run(pred,feed_dict=feeds)
        pred_test.append(predictions)
    print "Test loss:%.3f"%(avg_loss_test)
        # pirnt pic of validation
    np.savetxt('./pred_test_rnn/pred_test_epoch%3d'%epoch,pred_test)
    plt.clf()
    plt.title("Ground Truth and Predictions")
    plt.plot(y_validation[index_start:index_start + 50, 0], label="input")
    plt.plot(predictions[0:50, 0], ls='--', label="prediction")
    # plt.plot(y_validation[index_start:index_start+50,1], label="signal 1 (input)")
    # plt.plot(predictions[0:50,1], ls='--', label="signal 1 (prediction)")
    legend = plt.legend(frameon=True)
    legend.get_frame().set_facecolor('white')
    plt.draw()
    plt.pause(0.001)
    plt.savefig("./graph_rnn/GroundTruthandPredictions_%04i.jpg" % (epoch))

#not easy to shoulian, easy to stuck in local minimum


