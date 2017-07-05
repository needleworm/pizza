import pygame
import numpy as np

class Keyboard():

    def __init__(self,size = 128, length = 300):
        """
            size = midi tensor row size (default = 128)
        """


        print("Initializing Keyboard Module...")
        self.ascii2midi = dict([(113, 58), (50, 59), (119, 60), (101, 61), (52, 62), (114, 63), 
                    (53, 64), (116,65), (121,66), (55, 67), (117, 68), (56, 69), (105, 70),
                    (57,71), (111, 72), (112, 73)])
        self.motive = np.zeros((size,300),dtype=bool)
        self.line = np.zeros((size,1), dtype = bool)
        self.run = True
        self.size = size
        self.length = length
        self.sequence = 0
        pygame.init()
        self.clock = pygame.time.Clock()
     
    def _wait(self, time):

        return pygame.time.wait(time) 

    def _get(self):

        self.line = np.zeros((self.size, 1), dtype = bool)


        keys = pygame.key.get_pressed()
        for keyboard_index in range(len(keys)):
            if keys[keyboard_index] == 1:
                if keyboard_index in self.ascii2midi:
                    self.line[self.ascii2midi[keyboard_index],0] = True
       
        print(self.line.T)
        return self.line.T
    
    def _get_motive(self):

        self.run = True
        self.sequence = 0
        self.motive = np.zeros((self.size, self.length), dtype = bool)

        while self.run:

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: #esc 누르면 종
                        self.run = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.run = False
            


            if self.sequence == self.length:
                self.run = False
                break

            self.motive[:,self.sequence] = self._get()
            self.sequence = self.sequence + 1
            self._wait(60)

        return self.motive

    def start(self):


        """
           key board input start

           press ESC for quit
            

        """

        while self.run:

            self.tensor = np.zeros((self.size,1),dtype=bool)

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: #esc 누르면 종
                        self.run = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.run = False
            

            keys = pygame.key.get_pressed()
            for keyboard_index in range(len(keys)):
                if keys[keyboard_index] == 1:
                    if keyboard_index in self.ascii2midi:
                        self.tensor[self.ascii2midi[keyboard_index],0] = True
                        self.motive[self.ascii2midi[keyboard_index],self.sequence] = True
            
            pygame.time.wait(60)

        pygame.quit()
