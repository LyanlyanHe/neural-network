from mnist import MNIST
import time
import numpy as np

def data_loader(images):
    """Turn MNIST images arrays into numpy vectors arrays"""
    return [[[i / 255] for i in image] for image in images]

def data_parser(images, labels):
    return [(l, i) for i, l in zip(images, labels)]

s = time.time()
m = MNIST(".\datasets")
images, labels = m.load_training()
# data = data_parser(data_loader(images), labels)

print(time.time() - s)
s = time.time()
import random

random.shuffle(images)
print(time.time() - s)
# data = []
# def try_my_operation(i, l):
#     data = data_parser(data_loader(i), l)

# executor = concurrent.futures.ProcessPoolExecutor(2)
# f = [executor.submit(try_my_operation, item1, item2) for item1, item2 in zip(images, labels)]
# concurrent.futures.wait(f)
# print(data)
# print(time.time() - s, "seconds to initialised images")


# import pandas
# import time
# import mnist

# s = time.time()

# m = mnist.MNIST(".\datasets").load_training()

# print(time.time() - s)