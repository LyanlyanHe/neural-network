from network import Network
from draw import main, get_canvas
from numpy import argmax, array

net = Network([28 ** 2, 100, 10])
net.load_wb("wb.json")

def func():
    print(argmax(net.feedfoward(array([[x] for x in get_canvas()]))))

main(func)