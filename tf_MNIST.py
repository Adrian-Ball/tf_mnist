import tensorflow as tf 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#Number of nodes in each hidden layer
nodes_hl1 = 256
nodes_hl2 = 256
nodes_hl3 = 256
nodes_hl4 = 256
#Output classification labels
num_classes = 10 
#Number of images per training batch
batch_size = 100

#Input vector, 784 = 28 * 28
x = tf.placeholder('float', [None, 784])
#Output vector, represents digits 0-9, so len 10
y = tf.placeholder('float', [None, 10])

#Define the shape of the neural network. 
#Here, we have gone for 4 hidden layers betweem the input and output to keep things simple.
def nn_model(data):
    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([784, nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([nodes_hl1]))}

    hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([nodes_hl1, nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([nodes_hl2]))}

    hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([nodes_hl2, nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([nodes_hl3]))}

    hidden_layer_4 = {'weights':tf.Variable(tf.random_normal([nodes_hl3, nodes_hl4])),
                      'biases':tf.Variable(tf.random_normal([nodes_hl4]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([nodes_hl4, num_classes])),
                    'biases':tf.Variable(tf.random_normal([num_classes]))}

    l1 = tf.add(tf.matmul(data,hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3,hidden_layer_4['weights']), hidden_layer_4['biases'])
    l4 = tf.nn.relu(l3)

    output = tf.matmul(l4,output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    prediction = nn_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    num_epochs = 20
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):    
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',num_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

        #The above accuracy spits the dummy and complains of running out of memory, so we 
        #assess the accuracy in batches. We use the same batch size as training for simplicity.
        #c/o - https://github.com/tensorflow/tensorflow/issues/136
        # batch_num = int(mnist.test.num_examples / batch_size)
        # test_accuracy = 0
    
        # for i in range(batch_num):
        #     batch = mnist.test.next_batch(batch_size)
        #     test_accuracy += accuracy.eval(feed_dict={x: batch[0],
        #                                       y: batch[1]})
        #                                       #keep_prob: 1.0})
        # test_accuracy /= batch_num
        # print("test accuracy %g"%test_accuracy)

        #Find the first error and save it
        get_learned_results = tf.argmax(prediction, 1)
        learned_results = get_learned_results.eval({x:mnist.test.images})
        results_v_labels = correct.eval({x:mnist.test.images, y:mnist.test.labels})
        for i in range(len(mnist.test.labels)):
            if not results_v_labels[i]:
                plt.imsave('labelled_wrong.png', np.array(mnist.test.images[i]).reshape(28,28), cmap=cm.gray)
                file = open('wrong_image_reference','w') 
                file.write('Actual image label was:' + np.array2string(mnist.test.labels[i].argmax()) + '\n') 
                file.write('Learned label was:' + np.array2string(learned_results[i]) + '\n') 
                file.close() 
                break

train_neural_network(x)