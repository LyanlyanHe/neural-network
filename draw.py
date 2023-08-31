import math
import pygame 
import time

cell_size = 10
row = 28
column = 28

canvas = [0 for _ in range(28 * 28)] # list of tuple with relative coords 

screen = pygame.display.set_mode((cell_size * column, cell_size * row))

cell = pygame.surface.Surface((cell_size, cell_size))

background_color = (0, 0, 0)
 
def canvas_reload():
    global canvas
    canvas = [x if (0 <= x <= 1) else (1.0 if x > 1 else 0.0) for x in canvas]
    
    screen.fill(background_color)
    
    for index, intensity in enumerate(canvas):
        # print("h")intensity
        if intensity == 0:
            continue
        
        cell.fill(pygame.Color(255, 255, 255))
        cell.set_alpha(round(intensity * 255))
        # print((index % column, math.floor((index) / column), index))
        screen.blit(cell, (index % column * cell_size, math.floor(index / row) * cell_size))
    
    pygame.display.update()   

def set_canvas(c):
    global canvas
    canvas = c
    canvas_reload()

def get_canvas():
    return canvas


def canvas_reset():
    global canvas
    canvas = [0.0 for _ in range(row * column)]
    canvas_reload() 

def surrounding_coord(x, y):
    return [(x + j, y + i) for i in range(-1, 2) for j in range(-1, 2) if (
        not i == j == 0 and x + j >= 0 and x + j < column and y + i < row and y + i >= 0)]

def draw_cell(coord: tuple, erase=False):
    # print("welp")
    for s in surrounding_coord(coord[0], coord[1]):
        canvas[s[0] + s[1] * column] += 0.005 * (-1 if erase else 1)
    canvas[coord[0] + coord[1] * column] += 0.1 * (-1 if erase else 1)
    # print(surrounding_coord(coord[0], coord[1]))
    # print(canvas)
    canvas_reload()

# def erase_cell():
#     # mouse_coords = pygame.mouse.get_pos()
#     # canvas_coords = [math.floor(mouse_coords[0] / cell_size), math.floor(mouse_coords[1] / cell_size)]

#     mouse_color = tuple(list(screen.get_at([x * cell_size + int(cell_size / 2) for x in canvas_coords]))[:3])
#     if mouse_color not in canvas:
#         return
#     else:
#         color_coords = canvas[mouse_color]

#         color_coords.remove(canvas_coords)
#         if color_coords == []:
#             canvas.pop(mouse_color)
            
#         cell.fill(background_color)
        
#         if show_grid: 
#             pygame.draw.rect(cell, (0, 0, 0), (0, 0, cell_size, cell_size), 1)

#         screen.blit(cell, [x * cell_size for x in canvas_coords])
    

def main(func=lambda : None):
    global brush_color
    pygame.init()
    placing_cell = 0
    canvas_reset()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            
            if  event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    placing_cell = 1
                elif event.button == 3:
                    placing_cell = -1
            if event.type == pygame.MOUSEBUTTONUP:
                placing_cell = 0
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    canvas_reset()
                elif event.key == pygame.K_SPACE:
                    func()
 
            #     elif event.key == pygame.K_0:
            #         show_grid = False if show_grid else True
            #         canvas_reload()
            #     elif event.key in color_map:
            #         brush_color = color_map[event.key]

        if placing_cell != 0:
            # print("mouse press")
            mouse_coords = pygame.mouse.get_pos()
            # print(mouse_coords)
            canvas_coords = [math.floor(mouse_coords[0] / cell_size), math.floor(mouse_coords[1] / cell_size)]

            # print(canvas_coords)
            if (0 <= canvas_coords[0] < row and 0 <= canvas_coords[1] < column): draw_cell(canvas_coords, False if placing_cell == 1 else True)
            
        #     place_cell()
        # elif placing_cell == -1:
        #     remove_cell()

        

if  "__main__" == __name__:
    main()
    pygame.quit()

    