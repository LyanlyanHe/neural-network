from mnist import MNIST
import numpy as np
import time
import random
import json
from image_processor import *

class ActivationFunction:
    @staticmethod
    def sigmoid(x, d=False):
        return 1 / (1 + np.exp(-x)) if not d else ActivationFunction.sigmoid(x) * (1 - ActivationFunction.sigmoid(x))       

    @staticmethod
    def tnh(x, d=False):
        return np.tanh(x) if not d else 4 / (np.exp(x) + np.exp(-x)) ** 2

    @staticmethod
    def reLU(x, d=False): # rectified linear unit
        return max(0, x) if not d else 0 if x < 0 else 1

class CostFunction:
    @staticmethod # derivative for cost output ∂C/∂z
    def MSE(x, y, z=0, activationFunc=ActivationFunction.sigmoid, d=False):
        return  sum((x - y) ** 2) / len(x) if not d else 2 * (x - y) * activationFunc(z, d=True)

    @staticmethod
    def CrossEntropy(x, y, z=0, activationFunc=None, d=False):
        return - (y * np.log(x) + (1 - y) * np.log(1 - x)) / len(x) if not d else (x - y)
    
    @staticmethod 
    def SoftMax(x, z, y, activationFunc=np.log, d=False): # soft max take in weighted input rather den final activation values
        """ INCORRECT, SOFTMAX IS WRONG AND IDK WHY """
        return activationFunc(np.exp(z[y][0]) / sum([x[0] for x in np.exp(z)])) if not d else np.array(
            [[((sum([x[0] for x in np.exp(z)]) - np.exp(i[0])) / np.exp(i[0]))] for i in z]) 
    

class Network:
    def __init__(self, structure: list, activationFunc=ActivationFunction.sigmoid):
        self.structure = structure
        self.layers = len(structure)
        self.weights = [np.random.randn(y, x) / np.sqrt(x) 
                        for x, y in zip(structure[:-1], structure[1:])]
        self.bias = [np.random.randn(y, 1) for y in structure[1:]]
        self.activationFunc = activationFunc
    
    def feedfoward(self, a):
        x = a.copy()
        for w, b in zip(self.weights, self.bias):
            x = self.activationFunc(np.dot(w, x) + b)
        
        return x 
    
    def SGD(self, 
            training_data: list, 
            learning_rate: float, 
            mini_batch_size: int,
              epoch: int, 
              rp: float, 
              mp: float,
              testing_data=None,
              Cost=CostFunction.MSE,
              ):
        # data structure (label, images)
        
        for e in range(epoch):
            random.shuffle(training_data)
            
            nabla_w_p = [np.zeros(w.shape) for w in self.weights]
            nabla_b_p = [np.zeros(b.shape) for b in self.bias]
            for x in range(0, len(training_data), mini_batch_size):
                nabla_w = [np.zeros(w.shape) for w in self.weights]
                nabla_b = [np.zeros(b.shape) for b in self.bias]

                for label, image in training_data[x: x + mini_batch_size]:
                    delta_w, delta_b = self.backpropagation(image, label, Cost)

                    nabla_w = [nw + dw for nw, dw in zip(nabla_w, delta_w)]
                    nabla_b = [nb + db for nb, db in zip(nabla_b, delta_b)]
                
                self.weights = [w * (1 - (learning_rate * rp / len(training_data))) + mp * mw + (-learning_rate / mini_batch_size * nw) for nw, w, mw in zip(nabla_w, self.weights, nabla_w_p)]
                self.bias = [b + (-learning_rate / mini_batch_size * nb + mp * mb) for nb, b, mb in zip(nabla_b, self.bias, nabla_b_p)]

                nabla_w_p = nabla_w.copy()
                nabla_b_p = nabla_b.copy()


            print("Epoch {} Completed, accuracy is {}".format(e, self.evaluate(testing_data) * 100))
    
    def backpropagation(self, 
                        image,
                        label,
                        cost_func=CostFunction.MSE):
        z = []
        activations = [] 

        # feedforwarding
        a = image.copy()
        activations.append(image)
        for w, b in zip(self.weights, self.bias):
            # print(a.shape, w.shape, b.shape)
            a = np.dot(w, a) + b
            z.append(a.copy())
            a = self.activationFunc(a)
            activations.append(a)
        
        # propagating backwards
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.bias]

        # begins with the last layer
        delta = cost_func(x=a,y=np.array([[0 if label != i else 1] for i in range(10)]), z=z[-1], activationFunc=self.activationFunc,d=True)
        
        nabla_b[-1] = delta.copy()
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # propagate the rest of the layers
        for i in range(-2, -self.layers, -1):
            delta = np.dot(self.weights[i + 1].transpose(), delta) * self.activationFunc(z[i], d=True)
            nabla_b[i] = delta.copy()
            nabla_w[i] = np.dot(delta, activations[i - 1].transpose()) 
        
        return nabla_w, nabla_b

    def evaluate(self, test_data):
        return len([1 for label, image in test_data if label == np.argmax(self.feedfoward(image))]) / len(test_data)
    
    def save_wb(self, file_name):
        dict_ = {
            "weights": [],
            "bias": []
        }

        for w in self.weights:
            dict_["weights"].append(w.tolist())

        for b in self.bias:
            dict_["bias"].append(b.tolist())
        
        json.dump(dict_, open(file_name, "w"), indent=4)
    
    def load_wb(self, file_name):
        dict_ = json.load(open(file_name))

        self.weights.clear()
        self.bias.clear()

        for w in dict_["weights"]:
            self.weights.append(np.array(w))
        
        for b in dict_["bias"]:
            self.bias.append(np.array(b))
        
        # for w in self.weights:
        #     print(w.shape)        
        # for b in self.bias:
        #     print(b.shape)
        # print(len(dict_["weights"]))

        
if "__main__" == __name__:
    s = time.time()
    
    m = MNIST(".\datasets")
    images, label = m.load_training()
    t_images, t_label = m.load_training()
    
    # print(len(images[0]))
    # times = 3
    # for x in range(times):
    #     i_o = images.copy()
    #     i_o_f = []
    #     for image in i_o:
    #         i_o_f.append(twod2list(offset(random.randint(-5, 5), random.randint(-5, 5), list2_2d(28, image))))
    #     images.extend(i_o_f)
    # label *= times
        
    im = [np.array([[i / 255] for i in x]) for x in images]
    tim = [np.array([[i / 255] for i in x]) for x in t_images]
    # print(len(images))
    print(time.time() - s)
    net = Network([28 ** 2, 100, 10], ActivationFunction.sigmoid)
    net.load_wb("wb.json")
    net.SGD(list(zip(label, im)), 1, 100, 10, 5, .0025, list(zip(t_label, tim)), Cost=CostFunction.CrossEntropy)

    # print("offsetting phrase")
    # times = 6
    # for j in range(times):
    #     k = time.time()
    #     print("phrase " + str(j))
    #     oi = [twod2list(offset(random.randint(3, 5) * random.choice([-1, 1]), 
    #                            random.randint(3, 5) * random.choice([-1, 1]),
    #                            list2_2d(28, image))) for image in images]
    #     oi = [np.array([[i / 255] for i in x]) for x in oi]
    #     ti = [twod2list(offset(random.randint(3, 5) * random.choice([-1, 1]), 
    #                            random.randint(3, 5) * random.choice([-1, 1]),
    #                            list2_2d(28, image))) for image in t_images]
    #     ti = [np.array([[i / 255] for i in x]) for x in ti]
    #     print("image processed in ", time.time() - j)

    #     net.SGD(list(zip(label, oi)), 1.2, 100, 5, 5, .0001, list(zip(t_label, ti)), Cost=CostFunction.CrossEntropy)

    net.save_wb("wb.json")
    print((time.time() - s) / 60, " minutes taken" )

    # # net.feedfoward(data[0][1])
    # for w, b in zip(net.weights, net.bias):
    #     print(w.shape, b.shape)
    # net.load_wb("wb.json")
    # print(net.evaluate(list(zip(t_label, t_images))))


# print(ActivationFunction.sigmoid(0, d=True))
    

 
