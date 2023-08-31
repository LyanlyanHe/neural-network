import random

# images = []
# with open("training data", 'r') as file:
#     images = [(x[2:].rstrip("\n").split(), int(x[0])) for x in file.readlines()]

# max_offset_scale = 5

def list2_2d(col, l):
    return [[l[i + j] for j in range(col)] for i in range(0, len(l), col)]

def twod2list(l):
    return [y for x in l for y in x]

def offset(x, y, image):
    # X 
    if x > 0:
        for r in range(len(image)):
            image[r] = [0 for _ in range(x)] + image[r]
            
            for _ in range(x):
                image[r].pop(-1)
    elif x < 0:
        for r in range(len(image)):
            image[r] = image[r] + [0 for _ in range(abs(x))]

            for _ in range(abs(x)):
                image[r].pop(0)

    # Y 
    if y > 0:
        image.extend([[0 for _ in range(len(image[0]))] for _ in range(y)])
        for _ in range(y):
            image.pop(0)
    elif y < 0:
        image = [[0 for _ in range(len(image[0]))] for _ in range(abs(y))] + image
        for _ in range(abs(y)):
            image.pop(-1)
    
    return image