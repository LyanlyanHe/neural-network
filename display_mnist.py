from draw import main, set_canvas
from mnist import MNIST 
import random
from image_processor import *

mnist = MNIST(".\datasets")
images, label = mnist.load_training()


def func():
    i = random.randint(0, len(images) - 1)
    set_canvas(twod2list(offset(random.randint(3, 5) * random.choice([-1, 1]), 
                               random.randint(3, 5) * random.choice([-1, 1]),
                               list2_2d(28, [x / 255 for x in images[i]]))))
    print(label[i])

main(func=func)