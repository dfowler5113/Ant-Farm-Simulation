import random
import pygame 
import math
import torch.nn 
import torch.nn as nn
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer


pygame.init()
DISPLAY_WIDTH = 1500    
CLOSE_TO_BORDER_DISTANCE = 10
DISPLAY_HEIGHT = 800
screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))

# create a rectangle to represent the portion of the screen that will be displayed
viewport = screen.get_rect()

# initialize the mouse position
mouse_x, mouse_y = 0, 0


clock = pygame.time.Clock()
# Red color
red = pygame.Color(255, 0, 0)

# Brown color
brown = pygame.Color(165, 42, 42)

# Blue color
blue = pygame.Color(0, 0, 255)

MAX_MEMORY = 1000
BATCH_SIZE = 1
LR = 0.1


class Food(pygame.sprite.Sprite):
    def __init__(self,x,y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((3,3))
        self.image.fill((0, 255, 0))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
class Borders(pygame.sprite.Sprite):
    def __init__(self,x,y,name):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((15,15))
        self.image.fill((0, 255, 0))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.x = self.rect.x
        self.y = self.rect.y
        
        self.font = pygame.font.SysFont("Verdana", 20)
        self.text = self.font.render(name, True, (165, 42, 42))
        self.bg = pygame.Rect(x, y, 15, 15)
        self.pos1 = x,y
        self.pos = self.text.get_rect(center = (self.bg.x + self.bg.width/2,self.bg.y + self.bg.height/2))
        screen.blit(self.text, self.pos)

    def test(self,display):
        self.pos = self.text.get_rect(center = (self.bg.x + self.bg.width/2,
                                            self.bg.y + self.bg.height/2))
        display.blit(self.text, self.pos)
class Queen(pygame.sprite.Sprite):
        def __init__(self, x, y,color):
            pygame.sprite.Sprite.__init__(self)
            self.image = pygame.Surface((10,10))
            self.color = color
            self.image.fill((color))
            self.rect = self.image.get_rect()
            self.rect.x = x
            self.rect.y = y
            self.x = self.rect.x
            self.y = self.rect.y
            self.health = 500
            self.canMate = True
            
            self.font = pygame.font.SysFont("Verdana", 20)
            self.text = self.font.render("Queen", True, (165, 42, 42))
            self.bg = pygame.Rect(x, y, 15, 15)
            self.pos1 = x,y

            self.pos = self.text.get_rect(center = (self.bg.x + self.bg.width/2,self.bg.y + self.bg.height/2))
            screen.blit(self.text, self.pos)

        def test(self,display):
            self.pos = self.text.get_rect(center = (self.bg.x + self.bg.width/2,
                                                self.bg.y + self.bg.height/2))
            display.blit(self.text, self.pos)


class Ant(pygame.sprite.Sprite):
        def __init__(self, x, y,color,foodgroup, antgroup, number, QueenGroup):
            pygame.sprite.Sprite.__init__(self)
            #ant properties
            self.TypeOfAnt = random.randint(0,2)
            self.image = pygame.Surface((5,5))
            self.color = color
            self.image.fill((color))
            self.rect = self.image.get_rect()
            self.rect.x = x
            self.rect.y = y
            self.x = self.rect.x
            self.y = self.rect.y
            self.number = number
            self.antgroup = antgroup
            self.foodgroup = foodgroup
            self.QueenGroup = QueenGroup
            self.health = 100
            self.canMate = True
            self.CanAttack = False
            self.CanFarm = False
            self.numOfMoves = 0
            self.current_episode = []
            #learning properties
            self.n_moves = 0
            self.epsilon = 0 # randomness
            self.gamma = 0.0000009 # discount rate
            self.memory = deque(maxlen=MAX_MEMORY) # popleft()
            self.model = Linear_QNet(25, 256, 4)
            self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
            self.reward = 0
            self.up = False
            self.down = False
            self.left = False
            self.right = False
            self.antSpawned = False 
            
            self.queenreward = 0
            self.matingreward = 0
            self.farmerantreward = 0
            self.otherantseatreward =0
            self.farmerantreward =0
            self.fighterreward =0
            self.friendlyreward =0
            self.fighterattackqueenreward =0
            self.attackreward =0
            self.bumpqueenreward =0
            self.epsilon = 50
            self.bumpedqueen = False
            self.attacked = False
            self.attackedqueen = False
            self.mated = False
            self.foundfood = False
            self.bumpedfriend = False
           
            self.didanything = False

            #Farm ant
            if self.TypeOfAnt == 0:
                self.Storage = 2
                self.text_value = "Farm Ant" 
            #Fighter ant
            if self.TypeOfAnt == 1:
                self.Storage= 1
                self.text_value = "Fighter ant"      
            #Mating Ant
            if self.TypeOfAnt == 2:
                self.Storage= 1
                self.text_value = "Mating Ant"
                

            
            #Generate text that shows type
            self.font = pygame.font.SysFont("Verdana", 20)
            self.text = self.font.render(self.text_value, True, (165, 42, 42))
            self.bg = pygame.Rect(x, y, 15, 15)
            self.pos1 = x,y    

            #self.agentlearn()
        def get_surroundings(self):
            # Create a new surface for the ant's surroundings
            surroundings = pygame.Surface((6, 6))
            # Get the rect for this surface and center it around the ant's position
            rect = surroundings.get_rect(center=(self.rect.x, self.rect.y))

            # Initialize a 3x3 array to hold the information about the ant's surroundings
            grid = [[0 for _ in range(3)] for _ in range(3)]

            # Check each cell in the grid
            for i in range(3):
                for j in range(3):
                    # Calculate the position of this cell relative to the ant's position
                    cell_position = (rect.x + i - 1, rect.y + j - 1)

                    # Check if there's any food in this cell
                    for food in self.foodgroup:
                        if food.rect.collidepoint(cell_position):
                            grid[i][j] = 1  # Mark this cell as containing food

                    # Check if there's any ants in this cell
                    for ant in self.antgroup:
                        if ant.rect.collidepoint(cell_position):
                            if ant.color == self.color:
                                grid[i][j] = 2  # Mark this cell as containing an ant of the same color
                            else:
                                grid[i][j] = 3  # Mark this cell as containing an ant of a different color

                    # Check if there's any queens in this cell
                    for queen in self.QueenGroup:
                        if queen.rect.collidepoint(cell_position):
                            if queen.color == self.color:
                                grid[i][j] = 4  # Mark this cell as containing a queen of the same color
                            else:
                                grid[i][j] = 5  # Mark this cell as containing a queen of a different color
            #print(grid)
            return grid
        def distance_to_nearest_queen(self):
            #min_distance = float('inf')
            distance = 0
            for queen in self.QueenGroup:
                if queen.color == self.color:
                    distance = ((queen.rect.x - self.rect.x) ** 2 + (queen.rect.y - self.rect.y) ** 2) ** 0.5
            
            return distance
        def get_state(self):
            # Get the 3x3 grid around the ant
            surroundings = self.get_surroundings()
            
            coordinates = [self.rect.x, self.rect.y]
            pastmove = [self.up, self.left,self.right,self.down]
            reward = [self.reward]
            actions = [self.bumpedqueen,self.attacked,self.attackedqueen,self.mated,self.foundfood,self.bumpedfriend]
            didanything = [self.didanything]
            health = [self.health]
            #print(reward)
            # Flatten the grid to a single list
            flattened_surroundings = [cell for row in surroundings for cell in row]

            # Get the distance to the nearest queen of a different color
            distance_to_queen = self.distance_to_nearest_queen()

            # Combine the flattened grid and the distance to the queen into a single list
            state = flattened_surroundings + [distance_to_queen] + coordinates + pastmove + reward + actions + didanything + health

            return np.array(state, dtype=float)

        def remember(self, state, action, reward, next_state,done):
            self.memory.append((state, action, reward, next_state,done)) # popleft if MAX_MEMORY is reached

        

        def train_short_memory(self, state, action, reward, next_state,done):
            self.trainer.train_step(state, action, reward, next_state,done)

        def get_action(self, state):
            # random moves: tradeoff exploration / exploitation
            
            self.epsilon_decay = 1.00001  
            final_move = [0,0,0,0]
            
            if random.uniform(0, 100) < self.epsilon:
                move = random.randint(0, 3)
                final_move[move] = 1
                
            else:
                state0 = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()
                final_move[move] = 1
                
            #print(final_move)
            self.epsilon /= self.epsilon_decay
            print(self.epsilon)
            
            return final_move
        def train_long_memory(self):
            if len(self.memory) > BATCH_SIZE:
                mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
            else:
                mini_sample = self.memory

            states, actions, rewards, next_states, dones = zip(*mini_sample)
            self.trainer.train_step(states, actions, rewards, next_states, dones)
            #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)
        def agentlearn(self):
            # Define an episode length
            EPISODE_LENGTH = 5 # Adjust this as needed
            
            
            
            # Get the old state
            state_old = self.get_state()

            # Get the action
            final_move = self.get_action(state_old)

            # Perform the action and get the new state
            reward, done = self.play_step(final_move)
            state_new = self.get_state()

            # Add the step to the current episode
            self.current_episode.append((state_old, final_move, reward, state_new, done))
            #print(self.current_episode)
            # If the episode has reached the maximum length or the ant has hit a border, train the model based on the episode
            if len(self.current_episode) >= EPISODE_LENGTH :
                # Train short memory and remember for each step in the episode
                for state_old, final_move, reward, state_new, done in self.current_episode:
                    self.train_short_memory(state_old, final_move, reward, state_new, done)
                    self.remember(state_old, final_move, reward, state_new, done)

                # Train long memory
                self.train_long_memory()

            

                # Clear the current episode
                self.current_episode.clear()
            

        
        def play_step(self, action):
            MOVE_STEP = 1
            self.pos = self.text.get_rect(center = (self.bg.x + self.bg.width/2,self.bg.y + self.bg.height/2))
            # Predicted moves
            move_right = self.rect.x + MOVE_STEP
            move_left = self.rect.x - MOVE_STEP
            move_down = self.rect.y + MOVE_STEP
            move_up = self.rect.y - MOVE_STEP

            # Screen boundaries
            min_x = left.x  # left border
            max_x = right.x  # right border
            min_y = top.y  # top border
            max_y = bottom.y  # bottom border

            # Check each move to ensure it's within the screen boundaries
            if action[0] == 1 and move_right <= max_x: # Move right
                self.rect.x = move_right
                self.bg.x = move_right
            if action[1] == 1 and move_left >= min_x:   # Move left
                self.rect.x = move_left
                self.bg.x = move_left
            if action[2] == 1 and move_down <= min_y:  # Move down
                self.rect.y = move_down
                self.bg.y = move_down
            if action[3] == 1 and move_up >= max_y:  # Move up
                self.rect.y = move_up
                self.bg.y = move_up

            self.up = action[3] == 1
            self.down = action[2] == 1
            self.left = action[1] == 1
            self.right = action[0] == 1
            
                
                
            screen.blit(self.text, self.pos)
        
           
            
            #check if ant eats food and check if ants collides with bad ant or good ant
            #farmer ant collides with food 
            if self.TypeOfAnt == 0:
                collided = pygame.sprite.spritecollide(self,self.foodgroup,dokill = False) 
                if collided:
                    for x in collided:
                        x.kill()
                        self.farmerantreward=2
                        self.foundfood = True
                        self.health+=2
                        collided = []
                        break
                else:
                    self.reward-=10
            #other ant collides with food       
            if (self.TypeOfAnt == 1 or self.TypeOfAnt == 2):
                collided = pygame.sprite.spritecollide(self,self.foodgroup,dokill = False) 
                if collided:
                    for x in collided:
                        x.kill()
                        self.otherantseatreward=1
                        self.foundfood = True
                        self.health+=2
                        collided = []
                        break
                else:
                    self.reward-=100
            #fighter ant collides with another ant
            if self.TypeOfAnt == 1:
                collided = pygame.sprite.spritecollide(self,self.antgroup,dokill = False)
                if collided:
                    for x in collided:
                        if x.color!= self.color:
                            self.fighterreward=2
                            x.health-=7
                            self.attacked = True
                            if x.health<0:
                                x.kill()
                            collided = []
                            break
                else:
                    self.reward-=1
            #fighter ant collides with other queen
            if self.TypeOfAnt == 1:
                collided = pygame.sprite.spritecollide(self,self.QueenGroup,dokill = False)
                if collided:
                    for x in collided:
                        if x.color!= self.color:
                            self.fighterattackqueenreward=5
                            
                            x.health-=1
                            self.attackedqueen = True
                            if x.health<0:
                                x.kill()
                            collided = []
                            break
            #other ants collide with queen 
            if self.TypeOfAnt == 1 or self.TypeOfAnt == 0:
                collided = pygame.sprite.spritecollide(self,self.QueenGroup,dokill = False)
                if collided:
                    for x in collided:
                        if x.color== self.color:
                            self.bumpqueenreward=5
                            collided = []
                            self.bumpedqueen = True
                            break
            #other ant collides with other ant thats not the same
            if self.TypeOfAnt == 2 or self.TypeOfAnt == 0:
                collided = pygame.sprite.spritecollide(self,self.antgroup,dokill = False)
                if collided:
                    for x in collided:
                        if x.color!= self.color:
                            self.attackreward=1
                            x.health-=3
                            self.attacked = True
                            if x.health<0:
                                x.kill()
                            collided = []
                            break
            #other ant collides with other ant thats is the same
            if self.TypeOfAnt == 2 or self.TypeOfAnt == 0 or self.TypeOfAnt == 1:
                collided = pygame.sprite.spritecollide(self,self.antgroup,dokill = False)
                if collided:
                    for x in collided:
                        if x.color== self.color and x != self:
                            self.friendlyreward=2
                            collided = []
                            self.bumpedfriend = False
                            print(self.bumpedfriend)
                            break
                else:
                    self.friendlyreward-=1
            #mate ant collides with queen 
            if self.TypeOfAnt == 2 and not self.antSpawned:
                collided = pygame.sprite.spritecollide(self,self.QueenGroup,dokill = False)
                if collided:
                    for x in collided:
                        if x.color== self.color :
                            self.matingreward=10
                            for i in range(3):
                                a = random.randint(180, 1220)
                                y = random.randint(50, 640)
                                num = self.number +1
                                self.mated = False
                                AntGroup.add(Ant(a,y,self.color,FoodGroup,AntGroup,num,QueenGroup))
                            collided = []
                            self.canMate = False
                            self.antSpawned = True 
                            break
                else:
                    self.matingreward = -1
            for a in self.QueenGroup:
                distance = ((a.rect.y - self.rect.y)**2 + (a.rect.x - self.rect.x)**2)**0.5
                if distance < 1:
                    self.queenreward = 50
                else:
                    # Linear interpolation between 99 (for distance=1) and 0 (for distance=map_size)
                    self.queenreward = 50 * (1 - (distance - 1) / (1700 - 1))
            if self.bumpedqueen== False and self.attacked== False and self.attackedqueen== False and self.mated== False and self.foundfood== False and self.bumpedfriend == False:
                self.didanything = False
            else:
                self.didanything = True
            
            #print(self.didanything)
                        
            self.reward = (self.queenreward + self.matingreward + self.farmerantreward+self.otherantseatreward + self.farmerantreward + self.fighterreward +self.friendlyreward + self.fighterattackqueenreward +self.attackreward + self.bumpqueenreward + self.health)
            if self.didanything == False:
                self.reward -=50
                self.health -=.5
            
            #print(self.reward)
            
            #display.blit(self.text, self.pos)
            self.gameover = False
            if self.health<=0:
                self.gameover = True
            #print (self.reward)
            return (self.reward,self.gameover)

       
        
        


                    
                    
                 

FoodGroup = pygame.sprite.Group()
AntGroup = pygame.sprite.Group()
Borderss = pygame.sprite.Group()
QueenGroup = pygame.sprite.Group()
left =Borders(140,40,"top left") #180 is border, 170 == outside 190 inside, #180 moves x #50 moves y top border
right = Borders(1300,660,"bottom right")#1220 is border, 1000 == inside 1300 outside, moves x, 640 bottom border, #1220 moves x #640 moves y
top = Borders(140,660,"bottom left")
bottom = Borders(1330,40,"top right")
Borderss.add(left)
#Borderss.add(leftright)
Borderss.add(right)
Borderss.add(top)
Borderss.add(bottom)

for num in range(1000):
    a = random.randint(180, 1220)
    b = random.randint(50, 640)
    FoodGroup.add(Food(a,b))

for num in range(10):
    x = random.randint(180, 1220)
    y = random.randint(50, 640)
    AntGroup.add(Ant(x,y,red,FoodGroup,AntGroup,num,QueenGroup))

x = random.randint(180, 1220)
y = random.randint(50, 640)
QueenGroup.add(Queen(x,y,red))


for num in range(10):
    x = random.randint(180, 1220)
    y = random.randint(50, 640)
    AntGroup.add(Ant(x,y,brown,FoodGroup,AntGroup,num,QueenGroup))

x = random.randint(180, 1220)
y = random.randint(50, 640)
QueenGroup.add(Queen(x,y,brown))


for num in range(10):
    x = random.randint(180, 1220)
    y = random.randint(50, 640)
    AntGroup.add(Ant(x,y,blue,FoodGroup,AntGroup,num,QueenGroup))

x = random.randint(180, 1220)
y = random.randint(50, 640)
QueenGroup.add(Queen(x,y,blue))

while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        
    

    
    # Clear the screen
    screen.fill((255, 255, 255))
    FoodGroup.draw(screen)
    Borderss.draw(screen)
    QueenGroup.draw(screen)
    for queen in QueenGroup:
        queen.test(screen)
    for ant in AntGroup:
            # Perform a step and add it to the current episode
        ant.agentlearn() 
    AntGroup.draw(screen)
    for border in Borderss:
        border.test(screen)
    pygame.display.update()

    
   

