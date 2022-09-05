import numpy as np
np.random.seed(0)

def sigmoid(num):
    return 1 / ( 1 + np.exp(-num))

def error(num):
    if num == out_outputs[0]:
        return 1/2 * (0.3 - out_outputs[0])**2
    if num == out_outputs[1]:
        return 1 / 2 * (0.99 - out_outputs[1]) ** 2

def derivative_error(num):
    if num == out_outputs[0]:
        return out_outputs[0] - 0.3
    if num == out_outputs[1]:
        return out_outputs[1] - 0.99

def derivative_sigmoid(num):
    return num*(1-num)

inputs = np.random.rand(6)
weights_hidden_1 = np.random.rand(6,4)

b1 = 0.5
b2 = 0.3

net_hidden = np.dot(inputs,weights_hidden_1) + b1
out_hidden = sigmoid(net_hidden)

weight_outputs = np.random.rand(len(out_hidden), 2)

net_outputs = np.dot(out_hidden, weight_outputs) + b2
out_outputs = sigmoid(net_outputs)

output_error = np.array([ error(i) for i in out_outputs ])

learning_rate = 0.3
print('targets: [0.3 , 0.99]')
print(f'inputs: {inputs}')
print("\n")
print(f'weights before training: \n hidden layer weights \n {weights_hidden_1} \n and \n output weights \n {weight_outputs}')
print("\n")
print(f'output before training: {out_outputs}')



for i in range(5000):
    weights_o1 = weight_outputs[:, 0]
    weights_o2 = weight_outputs[:, 1]

    weights_h1 = weights_hidden_1[:,0]
    weights_h2 = weights_hidden_1[:,1]
    weights_h3 = weights_hidden_1[:,2]
    weights_h4 = weights_hidden_1[:,3]

    Derivative_o1 = np.array([derivative_error(out_outputs[0]) * derivative_sigmoid(out_outputs[0]) * n for n in out_hidden])
    Derivative_o2 = np.array([derivative_error(out_outputs[1]) * derivative_sigmoid(out_outputs[1]) * n for n in out_hidden])

    for wo1 in range(len(Derivative_o1)):
        weights_o1[wo1] -= learning_rate * Derivative_o1[wo1]

    for wo2 in range(len(Derivative_o2)):
        weights_o2[wo2] -= learning_rate * Derivative_o2[wo2]


    Derivative_h1 = np.array([ (derivative_error(out_outputs[0]) * derivative_sigmoid(out_outputs[0]) * weights_o1[0] +
                                derivative_error(out_outputs[1]) * derivative_sigmoid(out_outputs[1]) * weights_o2[0]) * derivative_sigmoid(out_hidden[0]) * inputs[n] for n in range(len(inputs))  ])

    Derivative_h2 = np.array(
        [(derivative_error(out_outputs[0]) * derivative_sigmoid(out_outputs[0]) * weights_o1[1] +
          derivative_error(out_outputs[1]) * derivative_sigmoid(out_outputs[1]) * weights_o2[
              1]) * derivative_sigmoid(out_hidden[1]) * inputs[n] for n in range(len(inputs))])

    Derivative_h3 = np.array(
        [(derivative_error(out_outputs[0]) * derivative_sigmoid(out_outputs[0]) * weights_o1[2] +
          derivative_error(out_outputs[1]) * derivative_sigmoid(out_outputs[1]) * weights_o2[
              2]) * derivative_sigmoid(out_hidden[2]) * inputs[n] for n in range(len(inputs))])

    Derivative_h4 = np.array(
        [(derivative_error(out_outputs[0]) * derivative_sigmoid(out_outputs[0]) * weights_o1[3] +
          derivative_error(out_outputs[1]) * derivative_sigmoid(out_outputs[1]) * weights_o2[
              3]) * derivative_sigmoid(out_hidden[3]) * inputs[n] for n in range(len(inputs))])

    for wh1 in range(len(Derivative_h1)):
        weights_h1[wh1] -= learning_rate * Derivative_h1[wh1]
    for wh2 in range(len(Derivative_h2)):
        weights_h2[wh2] -= learning_rate * Derivative_h1[wh2]
    for wh3 in range(len(Derivative_h3)):
        weights_h3[wh3] -= learning_rate * Derivative_h1[wh3]
    for wh4 in range(len(Derivative_h4)):
        weights_h4[wh4] -= learning_rate * Derivative_h1[wh4]

    weights_hidden_1 = np.array([ weights_h1, weights_h2, weights_h3,weights_h4 ]).T
    net_hidden = np.dot(inputs, weights_hidden_1) + b1
    out_hidden = sigmoid(net_hidden)

    weight_outputs = np.array([ weights_o1, weights_o2 ]).T
    net_outputs = np.dot(out_hidden, weight_outputs) + b2
    out_outputs = sigmoid(net_outputs)

print("\n")
print(f'weights AFTER training: \n hidden layer weights \n {weights_hidden_1} \n and \n output weights \n {weight_outputs}')
print(f'output AFTER training: {out_outputs}')
