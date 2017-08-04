import pygame
import numpy as np
import tensorflow as tf
from mingus.core import notes, chords
from mingus.containers import *
from mingus.midi import fluidsynth
from os import sys





SF2_1 = 'model1.sf2'
SF2_2 = 'model2.sf2'
fluidsynth.init(SF2_2)

#Constant
channel = 8
FADEOUT = 0.25
WELL_H = 100
WELL_W = 16
Block_H = 6 
Block_W = 30


class Keyboard():

    def __init__(self,size = 128, length = 256, multi = 3 ):
        """
            size = midi tensor row size (default = 128)
        """


        print("Initializing Keyboard Module...")
        self.ascii2midi = dict([(113, 58), (50, 59), (119, 60), (101, 61), (52, 62), (114, 63), 
                    (53, 64), (116,65), (121,66), (55, 67), (117, 68), (56, 69), (105, 70),
                    (57,71), (111, 72), (112, 73)])

        self.index = dict([(58, 0), (59, 1), (60, 2), (61, 3), (62, 4), (63, 5), (64, 6), (65, 7), (66, 8),
                            (67, 9), (68, 10), (69, 11), (70, 12), (71, 13), (72, 14), (73, 15)])

        self.code = ['A','A#','B','C','C#','D','D#','E','F','F#','G','G#','A','A#','B','C']
        self.octave = [2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4]

        self.motive = np.zeros((size,length), dtype=bool)
        self.line = np.zeros((size,1), dtype = bool)
        self.screen = np.zeros((100,16), dtype = bool)
        self.ai_screen = np.zeros((100,16), dtype = bool)
        self.run = True
        self.size = size
        self.length = length
        self.sequence = 0
        self.is_ai = False
       

        self.keyboard_state = np.zeros((1,16), dtype = bool)
        self.black_key=[1,4,6,9,11,13]
     


        pygame.init()
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode((480,700))
        pygame.draw.rect(self.display, (128,32,152),
                                        (0, WELL_H  * Block_H, 480, 30))




        """
        
        Image part 

        """
        # white keyboard update
        # 10 keys
        self.pressed_white_type1 = pygame.image.load('Asset/pressed/pressed_type1.png')
        self.pressed_white_type2 = pygame.image.load('Asset/pressed/pressed_type2.png')
        self.pressed_white_type3 = pygame.image.load('Asset/pressed/pressed_type3.png')
        self.pressed_white_type4 = pygame.image.load('Asset/pressed/pressed_type4.png')
        self.pressed_white_type5 = pygame.image.load('Asset/pressed/pressed_type5.png')
        self.pressed_white_type6 = pygame.image.load('Asset/pressed/pressed_type6.png')
        self.pressed_white_type7 = pygame.image.load('Asset/pressed/pressed_type7.png')
       
        self.unpressed_white_type1 = pygame.image.load('Asset/unpressed/unpressed_type1.png')
        self.unpressed_white_type2 = pygame.image.load('Asset/unpressed/unpressed_type2.png')
        self.unpressed_white_type3 = pygame.image.load('Asset/unpressed/unpressed_type3.png')
        self.unpressed_white_type4 = pygame.image.load('Asset/unpressed/unpressed_type4.png')
        self.unpressed_white_type5 = pygame.image.load('Asset/unpressed/unpressed_type5.png')
        self.unpressed_white_type6 = pygame.image.load('Asset/unpressed/unpressed_type6.png')
        self.unpressed_white_type7 = pygame.image.load('Asset/unpressed/unpressed_type7.png')

        self.pressed_white_type1 = pygame.transform.scale(self.pressed_white_type1,(48,88))
        self.pressed_white_type2 = pygame.transform.scale(self.pressed_white_type2,(48,88))
        self.pressed_white_type3 = pygame.transform.scale(self.pressed_white_type3,(48,88))
        self.pressed_white_type4 = pygame.transform.scale(self.pressed_white_type4,(48,88))
        self.pressed_white_type5 = pygame.transform.scale(self.pressed_white_type5,(48,88))
        self.pressed_white_type6 = pygame.transform.scale(self.pressed_white_type6,(48,88))
        self.pressed_white_type7 = pygame.transform.scale(self.pressed_white_type7,(48,88))

        self.unpressed_white_type1 = pygame.transform.scale(self.unpressed_white_type1,(48,88))
        self.unpressed_white_type2 = pygame.transform.scale(self.unpressed_white_type2,(48,88))
        self.unpressed_white_type3 = pygame.transform.scale(self.unpressed_white_type3,(48,88))
        self.unpressed_white_type4 = pygame.transform.scale(self.unpressed_white_type4,(48,88))
        self.unpressed_white_type5 = pygame.transform.scale(self.unpressed_white_type5,(48,88))
        self.unpressed_white_type6 = pygame.transform.scale(self.unpressed_white_type6,(48,88))
        self.unpressed_white_type7 = pygame.transform.scale(self.unpressed_white_type7,(48,88))

        # black keyboard update
        # 6 keys
        self.pressed_black = pygame.image.load('Asset/pressed/pressed_type8.png')
        self.unpressed_black = pygame.image.load('Asset/unpressed/unpressed_type8.png')

        self.pressed_black = pygame.transform.scale(self.pressed_black,(31,55))
        self.unpressed_black = pygame.transform.scale(self.unpressed_black,(31,55))

        self.padding_black = pygame.transform.scale(self.unpressed_black,(12,55))
        self.padding_black2 = pygame.transform.scale(self.unpressed_black,(20,55))
    
    def get_motive(self):

        return self.motive


    def play_input(self):

        self.tensor = np.zeros((self.size,1),dtype=bool)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: #esc 누르면 종
                    self.run = False
                    self.terminate()
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.run = False
                    self.terminate()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    self.is_ai = not(self.is_ai)
            

        keys = pygame.key.get_pressed()
        for keyboard_index in range(len(keys)):
            if keys[keyboard_index] == 1:
                if keyboard_index in self.ascii2midi:
                    self.tensor[self.ascii2midi[keyboard_index],0] = True
                    self.motive[self.ascii2midi[keyboard_index],self.sequence] = True
            
        self.screen = self.update(self.tensor)
        self.draw_screen(self.display, self.screen, self.is_ai)
        self.update_keyboard(self.keyboard_state)
            
        pygame.time.wait(60)

    def play_ai(self, tensor):

        dimens = np.shape(tensor)
        length = dimens[1]
        size = dimens[0]

        i = 0
        while i < length:
            pygame.event.get()
            pygame.display.flip()
            self.tensor = tensor[:,i]  
            self.tensor = np.reshape(self.tensor,(-1,1))
            self.screen = self.update(self.tensor)
            self.draw_screen(self.display, self.screen, self.is_ai)
            self.update_keyboard(self.keyboard_state)
            
            pygame.time.wait(60)
            i = i + 1

    def play_note(self):
        pass
        
    def terminate(self):

        pygame.quit()



    def update(self, line):
        """
            update keyboard & note state
        """

        old_screen = self.screen
        old_ai_screen = self.ai_screen
        old_motive = self.motive
        self.screen = np.zeros((100,16), dtype = bool)
        self.ai_screen = np.zeros((100,16), dtype = bool)
        self.motive = np.zeros((self.size, self.length), dtype = bool)



        x = 1
        while x < self.length:
            y = 0
            while y  < self.size:
                self.motive[y][x] = old_motive[y][x-1]
                y = y + 1
            x = x + 1

        y = 1
        while y < WELL_H:
            x = 0
            while x < WELL_W:
                self.screen[y][x] = old_screen[y-1][x]
                self.ai_screen[y][x] = old_ai_screen[y-1][x]
                x = x + 1
            y = y + 1

        
        for z in range(self.size):
            self.motive[z][0] = line[z][0]
            if z in self.index:   
                self.screen[0][self.index[z]] = line[z][0]
                self.ai_screen[0][self.index[z]] = self.is_ai
                self.keyboard_state[0][self.index[z]] = line[z][0]
        ##print(self.screen)


        return self.screen

    def draw_screen(self,display,screen,is_ai = False):

        self.display.fill((0, 0, 0))
        

        y = 0
        while y < WELL_H:
            x = 0
            while x < WELL_W:
                curr = screen[y][x]
                color = (156, 29, 176) if not(self.ai_screen[y][x]) else (34, 182, 0)
                if curr:
                    pygame.draw.rect(display, color,
                                        (x * Block_W, (WELL_H - 1 - y) * Block_H, Block_W, Block_H))
                x = x + 1
            y = y + 1
        pygame.display.flip()

    def draw_keyboard(self):

        # keyboard stat e update

        pressed_white_color = (150, 150, 150)
        pressed_black_color =  (66, 66, 66)
        unpressed_white_color = (255, 255, 255)
        unpressed_black_color = (0, 0, 0)


        pygame.draw.rect(self.display, (32,32,152),
                                        (0, WELL_H  * Block_H, 480, 12))
        x = 0
        while x < self.WELL_W:
            curr = self.keyboard_state[0][x]
            if curr:
                if x in self.black_key:
                    pygame.draw.rect(self.display, pressed_black_color,
                                        (x * Block_W, (WELL_H + 2) * Block_H, Block_W, 88))
                else:
                    pygame.draw.rect(self.display, pressed_white_color,
                                        (x * Block_W, (WELL_H + 2) * Block_H, Block_W, 88))
            else:
                if x in self.black_key:  
                    pygame.draw.rect(self.display, unpressed_black_color,
                                        (x * Block_W, (WELL_H + 2) * Block_H, Block_W, 88))

                else:  
                    pygame.draw.rect(self.display, unpressed_white_color,
                                        (x * Block_W, (WELL_H + 2) * Block_H, Block_W, 88))

            x = x + 1
        pygame.display.flip()

   
    def update_keyboard(self, keyboard_state=[]):

        """
        self.logo = pygame.image.load('linuxlogo.jpg')
        self.logo = pygame.transform.scale(self.logo,(50,50))
        self.display.blit(self.logo, (0,0))
        """
        white_keyboard_state = []
        black_keyboard_state = []

        for ind in range(len(keyboard_state[0])):
            if keyboard_state[0][ind]:
                fluidsynth.play_Note(Note(self.code[ind], self.octave[ind]), channel, 100)

            if ind in self.black_key:
                black_keyboard_state.append(keyboard_state[0][ind])
            else:
                white_keyboard_state.append(keyboard_state[0][ind])

      

        self.display.blit(self.padding_black,(0,(WELL_H+2) * Block_H))
        self.display.blit(self.padding_black2,(463,(WELL_H+2) * Block_H))
        x = 0

        while x < 10:
            if x in [2,9]:
                if white_keyboard_state[x]:
                    self.display.blit(self.pressed_white_type1, (48*x,(WELL_H+2) * Block_H))
                else:
                    self.display.blit(self.unpressed_white_type1, (48*x,(WELL_H+2) * Block_H))
            elif x in [3]:
                if white_keyboard_state[x]:
                    self.display.blit(self.pressed_white_type2, (48*x,(WELL_H+2) * Block_H))
                else:
                    self.display.blit(self.unpressed_white_type2, (48*x,(WELL_H+2) * Block_H))
            elif x in [0,7]:
                if white_keyboard_state[x]:
                    self.display.blit(self.pressed_white_type6, (48*x,(WELL_H+2) * Block_H))
                else:
                    self.display.blit(self.unpressed_white_type6, (48*x,(WELL_H+2) * Block_H))
            elif x in [5]:
                if white_keyboard_state[x]:
                    self.display.blit(self.pressed_white_type4, (48*x,(WELL_H+2) * Block_H))
                else:
                    self.display.blit(self.unpressed_white_type4, (48*x,(WELL_H+2) * Block_H))
            elif x in [8,1]:
                if white_keyboard_state[x]:
                    self.display.blit(self.pressed_white_type7, (48*x,(WELL_H+2) * Block_H))
                else:
                    self.display.blit(self.unpressed_white_type7, (48*x,(WELL_H+2) * Block_H))
            elif x in [6]:
                if white_keyboard_state[x]:
                    self.display.blit(self.pressed_white_type5, (48*x,(WELL_H+2) * Block_H))
                else:
                    self.display.blit(self.unpressed_white_type5, (48*x,(WELL_H+2) * Block_H))
            else:
                if white_keyboard_state[x]:
                    self.display.blit(self.pressed_white_type3, (48*x,(WELL_H+2) * Block_H))
                else:
                    self.display.blit(self.unpressed_white_type3, (48*x,(WELL_H+2) * Block_H))
            x = x + 1

        x = 0 
        while x < 6:

            if black_keyboard_state[x]:
                block = self.pressed_black
            else:
                block = self.unpressed_black
            if x == 0:
                self.display.blit(block, (40,(WELL_H+2) * Block_H))
            elif x == 1:
                self.display.blit(block, (127,(WELL_H+2) * Block_H))
            elif x == 2:
                self.display.blit(block, (185,(WELL_H+2) * Block_H))
            elif x == 3:
                self.display.blit(block, (267,(WELL_H+2) * Block_H))
            elif x == 4:
                self.display.blit(block, (323,(WELL_H+2) * Block_H))
            elif x == 5:
                self.display.blit(block, (373,(WELL_H+2) * Block_H))
            x = x + 1       
        pygame.display.flip()