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

        self.index = dict([(58, 0), (59, 1), (60, 2), (61, 3), (62, 4), (63, 5), (64, 6), (65, 7), (66, 8),
                            (67, 9), (68, 10), (69, 11), (70, 12), (71, 13), (72, 14), (73, 15)])

        self.motive = np.zeros((size,300),dtype=bool)
        self.line = np.zeros((size,1), dtype = bool)
        self.screen = np.zeros((100,16), dtype = bool)
        self.run = True
        self.size = size
        self.length = length
        self.sequence = 0
        pygame.init()
        self.clock = pygame.time.Clock()

        self.WELL_H = 100
        self.WELL_W = 16
        self.Block_H = 6 
        self.Block_W = 30
     
    def _wait(self, time):

        return pygame.time.wait(time) 

    def _get(self):

        self.line = np.zeros((self.size, 1), dtype = bool)


        keys = pygame.key.get_pressed()
        for keyboard_index in range(len(keys)):
            if keys[keyboard_index] == 1:
                if keyboard_index in self.ascii2midi:
                    self.line[self.ascii2midi[keyboard_index],0] = True
       
        #print(self.line.T)
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

        self.display = pygame.display.set_mode((480,600))


        while self.run:

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
            

            keys = pygame.key.get_pressed()
            for keyboard_index in range(len(keys)):
                if keys[keyboard_index] == 1:
                    if keyboard_index in self.ascii2midi:
                        self.tensor[self.ascii2midi[keyboard_index],0] = True
                        self.motive[self.ascii2midi[keyboard_index],self.sequence] = True
            
            self.screen = self.update(self.tensor)
            self.draw_screen()
            pygame.time.wait(60)

    def terminate(self):

        pygame.quit()


    def update(self,line):

        old_screen = self.screen
        self.screen = np.zeros((100,16), dtype = bool)

        y = 1
        while y < self.WELL_H:
            x = 0
            while x < self.WELL_W:
                self.screen[y][x] = old_screen[y-1][x]
                x = x + 1
            y = y + 1

        
        for z in range(self.size):
            if z in self.index:
                self.screen[0][self.index[z]] = line[z][0]
        print(self.screen)
        return self.screen

    def draw_screen(self):

        self.display.fill((0, 0, 0))
        y = 0
        while y < self.WELL_H:
            x = 0
            while x < self.WELL_W:
                curr = self.screen[y][x]
                if curr:
                    color = (255, 128, 0)
                    pygame.draw.rect(self.display, color,
                                        (x * self.Block_W, (self.WELL_H - 1 - y) * self.Block_H, self.Block_W, self.Block_H))
                #    pygame.draw.rect(self.display, color,
                #                     (x * self.Block_W, y * self.Block_H, self.Block_W, self.Block_H))
                x = x + 1
            y = y + 1


        pygame.display.flip()



keyboard = Keyboard()
keyboard.start()
