import pygame
import numpy as np

class Keyboard:

    def __init__(self,size=128):
        """
            size = midi tensor row size (default = 128)
        """


        print("Initializing Keyboard Module...")
        self.ascii2midi = dict([(113, 58), (50, 59), (119, 60), (101, 61), (52, 62), (114, 63), 
                    (53, 64), (116,65), (121,66), (55, 67), (117, 68), (56, 69), (105, 70),
                    (57,71), (111, 72), (112, 73)])
        self.motive = np.zeros((size,300),dtype=bool)

    def start():

        print("start")

    def merge():

        print("merge")

    def process():

        print("process")



 
pygame.init()
 

 
clock = pygame.time.Clock()
run = True


ascii2midi = dict([(113, 58), (50, 59), (119, 60), (101, 61), (52, 62), (114, 63), 
                    (53, 64), (116,65), (121,66), (55, 67), (117, 68), (56, 69), (105, 70),
                    (57,71), (111, 72), (112, 73)])

motive = np.zeros((128,300),dtype=bool)
sequence = 0
while run:
 
    count = 0
    #pygame.event.get() : 키를 눌렀을때 이벤트
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: #esc 누르면 종
                run = False
 
 
    #pygame.key.get_pressed() - 전체 키배열중 현재 눌려져있는 키를 bool형식의 튜플로 반환
    keys = pygame.key.get_pressed()
    #print(len(keys));
    for keyboard_index in range(len(keys)):
        if keys[keyboard_index] == 1:
            count = count + 1
            if keyboard_index in ascii2midi:
                motive[ascii2midi[keyboard_index],sequence] = True

    print(motive[:,sequence]) 
    
    if sequence == 299:
        run = False
    sequence = sequence + 1
    pygame.time.wait(60)
    #print(pygame.time.get_ticks())
    #clock.tick(16)
 
pygame.quit()
