# from GridBoard import *
import GridBoard as GB

class Gridworld:

    def __init__(self, size=4, mode='static'):
        if size >= 4:
            self.board = GB.GridBoard(size=size)
        else:
            print("Minimum board size is 4. Initialized to size 4.")
            self.board = GB.GridBoard(size=4)

        # Add pieces, positions will be updated later
        self.board.add_piece('Player', 'P', (0, 0))
        self.board.add_piece('Goal', '+', (1, 0))
        self.board.add_piece('Pit', '-', (2, 0))
        self.board.add_piece('Wall', 'W', (3, 0))

        # model select
        if mode == 'static':
            self.init_grid_static()
        elif mode == 'player':
            self.init_grid_player()
        else:
            self.init_grid_rand()

    # 정해진 초기 위치로 piece 정보 변경
    def init_grid_static(self):
        self.board.component_dict['Player'].pos = (0, 3) #Row, Column
        self.board.component_dict['Goal'].pos = (0, 0)
        self.board.component_dict['Pit'].pos = (0, 1)
        self.board.component_dict['Wall'].pos = (1, 1)
    
    
    # Initialize player in random location, but keep wall, goal and pit stationary
    def init_grid_player(self):
        # 위치 초기화
        self.init_grid_static()
        # player 위치 랜덤 설정
        self.board.component_dict['Player'].pos = GB.rand_pair(0, self.board.size)

        if (not self.validate_board()):
            self.init_grid_player()

    #Check if board is initialized appropriately (no overlapping pieces)
    #also remove impossible-to-win boards
    def validate_board(self):
        valid = True

        player = self.board.component_dict['Player']
        goal = self.board.component_dict['Goal']
        wall = self.board.component_dict['Wall']
        pit = self.board.component_dict['Pit']

        all_positions = [piece for name,piece in self.board.component_dict.items()]
        all_positions = [player.pos, goal.pos, wall.pos, pit.pos]
        if len(all_positions) > len(set(all_positions)):
            return False

        corners = [(0,0),(0,self.board.size), (self.board.size,0), (self.board.size,self.board.size)]
        #if player is in corner, can it move? if goal is in corner, is it blocked?
        if player.pos in corners or goal.pos in corners:
            val_move_pl = [self.validate_move('Player', addpos) for addpos in [(0,1),(1,0),(-1,0),(0,-1)]]
            val_move_go = [self.validate_move('Goal', addpos) for addpos in [(0,1),(1,0),(-1,0),(0,-1)]]
            if 0 not in val_move_pl or 0 not in val_move_go:
                #print(self.display())
                #print("Invalid board. Re-initializing...")
                valid = False
        return valid

    
    def validate_move(self, piece, addpos=(0, 0)):
        piece_pos = self.board.component_dict[piece].pos
        pit_pos = self.board.component_dict['Pit'].pos
        wall_pos = self.board.component_dict['Wall'].pos
        
        outcome = 0 #0 is valid, 1 invalid, 2 lost game
        new_pos = GB.add_tuple(piece_pos, addpos)
        
        # 벽으로 이동한 경우
        if new_pos == wall_pos:
            outcome = 1 
            
        # 게임 범위 밖으로 이동한 경우
        if max(new_pos) > (self.board.size-1) or min(new_pos) < 0:
            outcome = 1
        
        # pit 으로 이동한 경우
        if new_pos == pit_pos:
            outcome = 2
        return outcome


    #Initialize grid so that goal, pit, wall, player are all randomly placed
    def init_grid_rand(self):
        #height x width x depth (number of pieces)
        self.board.component_dict['Player'].pos = GB.randPair(0,self.board.size)
        self.board.component_dict['Goal'].pos = GB.randPair(0,self.board.size)
        self.board.component_dict['Pit'].pos = GB.randPair(0,self.board.size)
        self.board.component_dict['Wall'].pos = GB.randPair(0,self.board.size)

        if (not self.validateBoard()):
            #print('Invalid grid. Rebuilding..')
            self.init_grid_rand()

    
    def make_move(self, action):
        #need to determine what object (if any) is in the new grid spot the player is moving to
        #actions in {u,d,l,r}
        # def check_move(addpos):
        #     if self.validate_move('Player', addpos) in [0, 2]:
        #         new_pos = GB.add_tuple(self.board.component_dict['Player'].pos, addpos)
        #         self.board.movePiece('Player', new_pos)

        if action == 'u': #up
            add_pos = (-1, 0)
        elif action == 'd': #down
            add_pos = (1, 0)
        elif action == 'l': #left
            add_pos = (0, -1)
        elif action == 'r': #right
            add_pos = (0, 1)
        else:
            raise ValueError('invalid action')
        
        # 정상적 이동인 경우 player 를 이동
        if self.validate_move('Player', add_pos) in [0, 2]:
            new_pos = GB.add_tuple(self.board.component_dict['Player'].pos, add_pos)
            self.board.move_piece('Player', new_pos)

    def reward(self):
        player_pos = self.board.component_dict['Player'].pos
        pit_pos = self.board.component_dict['Pit'].pos
        goal_pos = self.board.component_dict['Goal'].pos
        if (player_pos == pit_pos):
            return -10
        elif (player_pos == goal_pos):
            return 10
        else:
            return -1

    def display(self):
        return self.board.render()