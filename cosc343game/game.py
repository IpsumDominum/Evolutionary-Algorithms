import importlib
import numpy as np
import traceback
import sys
import gzip, pickle
from datetime import datetime
import os
import getopt
import signal

in_tournament = False
game_play = True

def alarm_handler(signum, frame):
    raise RuntimeError("Time out")

# Class avatar is a wrapper for the agent with extra bits required
# for runnin the game
class Avatar:

    # Initialise avatar for an agent of a given player
    def __init__(self,agent,player):
        self.agent = agent
        self.player = player
        self.position = np.zeros((2)).astype('int')
        self.next_position = np.zeros((2)).astype('int')
        self.reset_for_new_game()
        self.reset_for_new_turn()

    # Reset the avatar variables for a new game
    def reset_for_new_game(self):
        self.size = 1
        self.alive = 1
        self.energy = 2.0
        self.turn = 0
        self.strawb_eats = 0
        self.enemy_eats = 0.0

    # Reset the avater variables for a new turn in a game
    def reset_for_new_turn(self):
        self.bounce = False
        self.attackers = list()
        self.attacks = list()
        self.next_position *= 0
        self.size = int(np.floor(np.log2(self.energy)))

    # Execute AgentFunction that maps percepts to actions
    def action(self, percepts):

        if in_tournament:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(1)

        try:
            actions = self.agent.AgentFunction(percepts)
            if type(actions) == np.ndarray:
                actions = actions.tolist()

        except Exception as e:
            if in_tournament:
                raise RuntimeError("Error! Failed to execute AgentFunction")
            else:
                print("Error! Failed to execute AgentFunction from '%s.py'" % self.playerFile)
                traceback.print_exc()
                sys.exit(-1)

        if in_tournament:
            signal.alarm(0)

        if type(actions) != list:
            if in_tournament:
                raise RuntimeError("Error! AgentFunction must return a list or numpy.ndarray type")
            else:
                print("Error! AgentFunction in '%s.py' must return a list or numpy.ndarray type" % self.playerFile)
                traceback.print_exc()
                sys.exit(-1)

        if len(actions) != 7:
            if in_tournament:
                raise RuntimeError("Error! The returned action list from AgentFunction must contain 7 items")
            else:
                print("Error! The returned action list from AgentFunction in '%s.py'must contain 7 items" % self.playerFile)
                traceback.print_exc()
                sys.exit(-1)

        return actions

# Class player holds all the agents for a given player
class Player:

    def __init__(self, player, playerFile, nAgents,emptyMode=False):

        self.player = player
        self.nAgents = nAgents
        self.playerFile = playerFile
        self.fitness = list()
        self.errorMsg = ""
        self.ready = False

        if emptyMode:
            return

        # Import agent file as module
        if in_tournament:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(10)
        try:
            self.exec = importlib.import_module(playerFile)
        except Exception as e:
            if in_tournament:
                signal.alarm(0)
                self.errorMsg = str(e)
                return
            else:
                print("Error! Failed to load '%s.py'" % self.playerFile)
                traceback.print_exc()
                sys.exit(-1)

        if in_tournament:
            signal.alarm(0)

        if in_tournament:
            self.name = playerFile.split('.')[1]
        else:
            if hasattr(self.exec, 'playerName'):
                self.name = self.exec.playerName
            else:
                self.name = playerFile

        # Create the initial population of agents by creating
        # new instance of the agent using provided MyCreature class
        agents = list()
        for n in range(self.nAgents):
            if in_tournament:
                signal.signal(signal.SIGALRM, alarm_handler)
                signal.alarm(1)
            try:
                agent = self.exec.MyCreature()
            except Exception as e:
                if in_tournament:
                    signal.alarm(0)
                    self.errorMsg = str(e)
                    return
                else:
                    print("Error! Failed to instantiate MyCreature() from '%s.py'" % self.playerFile)
                    traceback.print_exc()
                    sys.exit(-1)

            if in_tournament:
                signal.alarm(0)
            agents.append(agent)

        # Convert list of agents to list of avatars
        try:
            self.agents_to_avatars(agents)
        except Exception as e:
            if in_tournament:
                signal.alarm(0)
                self.errorMsg = str(e)
                return
            else:
                print("Error! Failed to create a list of MyCratuers")
                traceback.print_exc()
                sys.exit(-1)

        self.ready = True


    def reset_for_new_game(self):
        for avatar in self.avatars:
            avatar.reset_for_new_game()

    # Convert list of agents to list of avatars
    def agents_to_avatars(self, agents):
        self.avatars = list()
        self.stats = list()

        for agent in agents:
            if type(agent) != self.exec.MyCreature:
                if in_tournament:
                    raise RuntimeError(
                        'Error! The new_population returned form newGeneration() must contain objects of MyCreature() type')
                else:
                    print("Error! The new_population returned form newGeneration() in '%s.py' must contain objects of MyCreature() type" %
                    self.playerFile)
                    traceback.print_exc()
                    sys.exit(-1)

            avatar = Avatar(agent,player=self)
            self.avatars.append(avatar)
            self.stats.append(dict())

    # Get a new generation of agents
    def new_generation_agents(self):

        # Record game stats in the agent objects
        old_population = list()
        for avatar in self.avatars:
            agent = avatar.agent
            agent.alive = avatar.alive
            agent.turn = avatar.turn
            agent.size = avatar.size
            agent.energy = avatar.energy
            agent.strawb_eats = avatar.strawb_eats
            agent.enemy_eats = avatar.enemy_eats
            old_population.append(agent)

        sys.stdout.write("  %s avg_fitness: " % self.name)
        sys.stdout.flush()

        # Get a new population of agents by calling
        # the provided newGeneration method
        if in_tournament:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(2)

        try:
            result = self.exec.newGeneration(old_population)
        except Exception as e:
            if in_tournament:
                raise RuntimeError('Error! Failed to execute newGeneration()')
            else:
                print("Error! Failed to execute newGeneration() from '%s.py'" % self.playerFile)
                traceback.print_exc()
                sys.exit(-1)

        if in_tournament:
            signal.alarm(0)

        if type(result) != tuple or len(result) != 2:
            if in_tournament:
                raise RuntimeError('Error! The returned value form newGeneration() must be a 2-item tuple')
            else:
                print("Error! The returned value form newGeneration() in '%s.py' must be a 2-item tuple" % self.playerFile)
                traceback.print_exc()
                sys.exit(-1)

        (new_population, fitness) = result

        if type(new_population) != list:
            if in_tournament:
                raise RuntimeError('Error! The new_population returned form newGeneration() must be a list')
            else:
                print("Error! The new_population returned form newGeneration() in '%s.py' must be a list" % self.playerFile)
                traceback.print_exc()
                sys.exit(-1)

        try:
            fitness = float(fitness)
        except Exception as e:
            if in_tournament:
                raise RuntimeError('Error! The fitness returned form newGeneration() must be float or int')
            else:
                print("Error! The new_population returned form newGeneration() in '%s.py' must be a float or int" % self.playerFile)
                traceback.print_exc()
                sys.exit(-1)

        if len(new_population) != self.nAgents:
            if in_tournament:
                raise RuntimeError('Error! The new_population returned form newGeneration() must contain %d items' % self.nAgents)
            else:
                print("Error! The new_population returned form newGeneration() in '%s.py' must contain %d items" % (self.playerFile, self.nAgents))
                traceback.print_exc()
                sys.exit(-1)

        sys.stdout.write(" %.2e\n" % fitness)
        sys.stdout.flush()
        self.fitness.append(fitness)

        # Convert agents to avatars
        self.agents_to_avatars(new_population)

# Class that runs the entire game
class Game:

    # Initialises the game
    def __init__(self, gridSize, nTurns, nAgents, nWalls, nGames,tournament=False):
        global in_tournament

        self.rnd = np.random.RandomState()
        self.gridSize = gridSize
        self.nTurns = nTurns
        self.nAgents = nAgents
        self.nGames = nGames
        self.nPercepts = 75
        self.nActions = 7
        self.nFood = self.nAgents
        self.nWalls = nWalls
        in_tournament = tournament


    def init_wall_map(self):
        # Create wall map - this stays constant for all games
        self.wall_map = np.zeros((self.gridSize, self.gridSize), dtype='int8')

        range_mid = self.gridSize / 2 - 0.5
        init_range_low = int(self.gridSize % 2)
        init_range_high = int(self.gridSize / 2)

        for n in range(int(self.nWalls/2)):
            while True:
                rand_offset = self.rnd.randint(init_range_low, init_range_high, size=(2)).astype('float')
                if self.gridSize % 2 == 0:
                    rand_offset += 0.5

                rand_flip = 2 * self.rnd.randint(0, 2, size=(2)) - 1

                rand_offset *= rand_flip

                position_wall_1 = np.array(
                    [int(range_mid + rand_offset[0]), int(range_mid + rand_offset[1])]).astype('int')
                position_wall_2 = np.array(
                    [int(range_mid - rand_offset[0]), int(range_mid - rand_offset[1])]).astype('int')

                x_w1 = position_wall_1[0]
                y_w1 = position_wall_1[1]
                x_w2 = position_wall_2[0]
                y_w2 = position_wall_2[1]

                if self.wall_map[x_w1, y_w1] != 0 or self.wall_map[x_w2, y_w2] != 0:
                    continue

                self.wall_map[x_w1, y_w1] = 1
                self.wall_map[x_w2, y_w2] = 1
                break

    # Create the new agent and food map - new for every epoch
    def init_agent_and_food_maps(self):
        self.agent_map = np.ndarray((self.gridSize, self.gridSize), dtype=Player)
        self.food_map = np.zeros((self.gridSize, self.gridSize), dtype='int8')

        range_mid = self.gridSize / 2 - 0.5
        init_range_low = int(self.gridSize % 2)
        init_range_high = int(self.gridSize / 2)

        for n in range(self.nAgents):
            while True:
                rand_offset = self.rnd.randint(init_range_low, init_range_high, size=(2)).astype('float')
                if self.gridSize % 2 == 0:
                    rand_offset += 0.5

                rand_flip = 2 * self.rnd.randint(0, 2, size=(2)) - 1

                rand_offset *= rand_flip

                position_player_1 = np.array([int(range_mid + rand_offset[0]), int(range_mid + rand_offset[1])]).astype('int')
                position_player_2 = np.array([int(range_mid - rand_offset[0]), int(range_mid - rand_offset[1])]).astype('int')

                x_p1 = position_player_1[0]
                y_p1 = position_player_1[1]
                x_p2 = position_player_2[0]
                y_p2 = position_player_2[1]

                if self.wall_map[x_p1,y_p1] != 0 or self.wall_map[x_p2,y_p2] != 0:
                    continue

                # Check that the agents are not positions too close by
                tryAgain = False
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        n_x = (x_p1 + i) % self.gridSize
                        n_y = (y_p1 + j) % self.gridSize

                        if (self.agent_map[n_x, n_y] != None):
                            tryAgain = True
                            break

                        n_x = (x_p2 + i) % self.gridSize
                        n_y = (y_p2 + j) % self.gridSize

                        if (self.agent_map[n_x, n_y] != None):
                            tryAgain = True
                            break

                    if tryAgain:
                        break

                if not tryAgain:
                    break

            self.agent_map[x_p1, y_p1] = self.players[0].avatars[n]
            self.players[0].avatars[n].position = np.array([x_p1, y_p1])

            self.agent_map[x_p2, y_p2] = self.players[1].avatars[n]
            self.players[1].avatars[n].position = np.array([x_p2, y_p2])

        for n in range(self.nAgents):
            while True:
                rand_offset = self.rnd.randint(init_range_low, init_range_high, size=(2)).astype('float')
                if self.gridSize % 2 == 0:
                    rand_offset += 0.5

                rand_flip = 2 * self.rnd.randint(0, 2, size=(2)) - 1

                rand_offset *= rand_flip

                position_food_1 = np.array([int(range_mid + rand_offset[0]), int(range_mid + rand_offset[1])]).astype('int')
                position_food_2 = np.array([int(range_mid - rand_offset[0]), int(range_mid - rand_offset[1])]).astype('int')

                x_f1 = position_food_1[0]
                y_f1 = position_food_1[1]
                x_f2 = position_food_2[0]
                y_f2 = position_food_2[1]

                if self.wall_map[x_f1, y_f1] != 0 or self.wall_map[x_f2,y_f2] != 0:
                    continue

                if self.agent_map[x_f1,y_f1] != None or self.agent_map[x_f2,y_f2] != None:
                    continue

                self.food_map[x_f1,y_f1] = 1
                self.food_map[x_f2,y_f2] = 1
                break

    # Update the agent map base on positions of agent after a
    def update_agent_map(self):
        self.agent_map = np.ndarray((self.gridSize, self.gridSize), dtype=Player)

        for p in range(2):
            for avatar in self.players[p].avatars:
                if not avatar.alive:
                    continue

                x = avatar.position[0]
                y = avatar.position[1]

                self.agent_map[x,y]=avatar

    # Update the stats for the visualiser
    def update_vis_agents(self,creature_state):
        for p in range(2):
            for n in range(self.nAgents):
                i = n + p * self.nAgents
                avatar = self.players[p].avatars[n]

                creature_state[i, 0] = avatar.position[0]
                creature_state[i, 1] = avatar.position[1]
                creature_state[i, 2] = avatar.alive
                creature_state[i, 3] = p
                creature_state[i, 4] = avatar.size

    # Run the game
    def run(self,player1File, player2File,show_games, save_games, visResolution=(720,480), visSpeed='normal',savePath="saved"):

        global in_tournament, game_play

        self.players = list()

        self.game_messages = ['', '']
        self.game_scores = [0, 0]
        self.game_saves = list()

        # Load player 1
        try:
            self.players.append(Player(0,player1File, self.nAgents))
        except Exception as e:
            if in_tournament:
                self.players.append(Player(0,player1File,self.nAgents,emptyMode=True))
                self.game_messages[0] = "Error! Failed to create a player with the provided MyAgent.py code"
            else:
                print('Error! ' + str(e))
                sys.exit(-1)

        if not self.players[0].ready:
            self.game_scores[0] = -self.nAgents
            if self.players[0].errorMsg != "":
                self.game_messages[0] = self.players[0].errorMsg
            game_play = False

        # Load player 2
        try:
            self.players.append(Player(1,player2File, self.nAgents))
        except Exception as e:
            if in_tournament:
                self.players.append(Player(1,player2File,self.nAgents,emptyMode=True))
                self.game_messages[1] = "Error! Failed to create a player with the provided MyAgent.py code"
            else:
                print('Error! ' + str(e))
                sys.exit(-1)

        if not self.players[1].ready:
            self.game_scores[1] = -self.nAgents
            if self.players[1].errorMsg != "":
                self.game_messages[1] = self.players[0].errorMsg
            game_play = False


        if not game_play:
            return

        # Create the visualiser
        if len(show_games)>0:
            import vis_pygame as vis
            self.vis = vis.visualiser(speed=visSpeed,gridSize=self.gridSize, playerStrings=(self.players[0].name,self.players[1].name),
                                  resolution=visResolution)

        # Initialise agents on new grid
        self.init_wall_map()

        vis_walls = list()
        for x in range(self.gridSize):
            for y in range(self.gridSize):
                if self.wall_map[x, y] == 1:
                    vis_walls.append((x, y))

        # Play the game a number of times
        for game in range(1,self.nGames+1):

            sys.stdout.write("\nGame %3d/%d..." % (game, self.nGames))
            sys.stdout.flush()

            # Reset avatars for a new game
            for p in range(2):
                self.players[p].reset_for_new_game()

            #Initialise agents on new grid
            self.init_agent_and_food_maps()

            #self.comp_vis_state(creature_state, creature_prev_state)
            vis_agents = np.zeros((self.nAgents * 2, 5, self.nTurns + 1)).astype('int')
            vis_food = list()

            if game in show_games:
                self.vis.reset()

            # Play the game over a number of turns
            for turn in range(self.nTurns):

                food_array = list()
                for x in range(self.gridSize):
                    for y in range(self.gridSize):
                        if self.food_map[x, y] == 1:
                            food_array.append((x, y))
                vis_food.append(food_array)
                self.update_vis_agents(vis_agents[:,:,turn])

                if game in show_games:
                    self.vis.show(creature_state=vis_agents[:,:,turn],food_array=vis_food[turn],wall_array=vis_walls,game=game, turn=turn)

                perceptBlock = 5
                pBHalf = int(perceptBlock / 2)

                # Create new agent map based on actions
                new_agent_map = np.ndarray((self.gridSize,self.gridSize), dtype=object)

                # Get actions of the agents
                for p in range(2):
                    for avatar in self.players[p].avatars:

                        if not avatar.alive:
                            continue

                        avatar.turn = turn+1
                        avatar.reset_for_new_turn()

                        position = avatar.position

                        # Percepts
                        percepts = np.zeros((perceptBlock,perceptBlock,3))

                        # Add nearby agents to percepts
                        for i in range(-pBHalf,pBHalf):
                            for j in range(-pBHalf,pBHalf):
                                x = (position[0] + i)%self.gridSize
                                y = (position[1] + j)%self.gridSize

                                if self.food_map[x,y] != 0:
                                   percepts[pBHalf + i, pBHalf + j, 1] = self.food_map[x,y]

                                if self.wall_map[x,y] != 0:
                                    percepts[pBHalf + i, pBHalf + j, 2] = self.wall_map[x, y]

                                if i==0 and j==0:
                                    percepts[pBHalf, pBHalf,0] = avatar.size
                                    continue

                                other_agent = self.agent_map[ x, y]

                                if other_agent is not None:
                                    if other_agent.player.player == p:
                                        s = 1
                                    else:
                                        s = -1

                                    percepts[pBHalf+i,pBHalf+j,0] = s*other_agent.size

                        # Get action from agent
                        try:
                            action = np.argmax(avatar.action(percepts))
                        except Exception as e:
                            if in_tournament:
                                self.game_scores[p] = -self.nAgents
                                self.game_messages[p] = str(e)
                                game_play = False
                            else:
                                traceback.print_exc()
                                sys.exit(-1)

                        if not game_play:
                            break

                        # Action 6 is a random movement
                        if action==6:
                            action = self.rnd.randint(0,4)

                        x = avatar.position[0]
                        y = avatar.position[1]

                        # Action 0 is move left
                        if action == 0:
                            x -= 1
                        # Action 1 is move up
                        elif action == 1:
                            y -= 1
                        # Action 2 is move right
                        elif action == 2:
                            x += 1
                        # Action 3 is move down
                        elif action == 3:
                            y += 1

                        # Action 4 is do nothing

                        x %= self.gridSize
                        y %= self.gridSize

                        # Action 5 is eat
                        if action==5 and self.food_map[x,y] != 0:
                            self.food_map[x, y] = 0
                            avatar.energy += 1
                            avatar.strawb_eats += 1

                        # Can't walk onto the wall
                        if self.wall_map[x,y] != 0:
                            x = avatar.position[0]
                            y = avatar.position[1]

                        if new_agent_map[x,y] is None:
                            new_agent_map[x,y] = list()

                        new_agent_map[x,y].append(avatar)
                        avatar.next_position[0]= x
                        avatar.next_position[1]= y

                if not game_play:
                    return

                # Check for agents bouncing (going into field occupied by other agents)
                for x in range(self.gridSize):
                    for y in range(self.gridSize):
                        if new_agent_map[x,y] is not None:
                            if len(new_agent_map[x,y])>1 or self.agent_map[x, y] is not None:
                               for avatar in new_agent_map[x,y]:
                                   avatar.bounce = True

                # Check for attacks
                for p in range(2):
                    for avatar in self.players[p].avatars:
                        if not avatar.alive:
                            continue

                        x = avatar.position[0]
                        y = avatar.position[1]
                        if new_agent_map[x,y] is not None:
                            for other_avatar in new_agent_map[x,y]:
                                if other_avatar != avatar:
                                    avatar.attackers.append(other_avatar)
                                    other_avatar.attacks.append(avatar)

                # Resolve attacks
                dead_avatars = list()
                for p in range(2):
                    for n in range(self.nAgents):
                        avatar = self.players[p].avatars[n]
                        if not avatar.alive:
                            continue

                        attack_size = -avatar.energy
                        enemy_attackers = 0
                        for attacker in avatar.attackers:
                            if attacker.player == avatar.player:
                                attack_size -= attacker.energy
                            else:
                                enemy_attackers += 1
                                attack_size += attacker.energy

                        if attack_size > 0:
                            booty_share = float(avatar.energy)/float(enemy_attackers)
                            for attacker in avatar.attackers:
                                if attacker.player != avatar.player:
                                   attacker.energy += booty_share
                                   attacker.enemy_eats += booty_share

                            dead_avatars.append(avatar)

                # Mark any as not alive if successfully attacked by the enemy
                for avatar in dead_avatars:
                    avatar.alive = False

                # Move the agents that performed a legal movement
                for p in range(2):
                    for avatar in self.players[p].avatars:
                        if not avatar.alive:
                            continue

                        if not avatar.bounce:
                            avatar.position[0] = avatar.next_position[0]
                            avatar.position[1] = avatar.next_position[1]

                # Update the agent map according to new positions
                self.update_agent_map()

                # Check for winner
                for p in range(2):
                    gameOver = True
                    for avatar in self.players[p].avatars:
                        if avatar.alive:
                           gameOver = False
                           break
                    if gameOver:
                        break
                if gameOver:
                    break


            food_array = list()
            for x in range(self.gridSize):
                for y in range(self.gridSize):
                    if self.food_map[x, y] == 1:
                        food_array.append((x, y))
            vis_food.append(food_array)
            self.update_vis_agents(vis_agents[:,:,turn+1])
            vis_agents = vis_agents[:,:,:turn+2]

            if game in show_games:
                self.vis.show(creature_state=vis_agents[:,:,turn+1],food_array=vis_food[turn+1],wall_array=vis_walls,game=game,turn=turn+1)


            survivorCount = np.zeros((2)).astype('int')
            for p in range(2):
                for avatar in self.players[p].avatars:
                    if avatar.alive:
                        survivorCount[p] += 1

            if survivorCount[0]>survivorCount[1]:
                sys.stdout.write('won by %s (blue) %d-%d\n' % (self.players[0].name,survivorCount[0],survivorCount[1]))
            elif survivorCount[1]>survivorCount[0]:
                sys.stdout.write('won by %s (red) %d-%d\n' % (self.players[1].name,survivorCount[1],survivorCount[0]))
            else:
                sys.stdout.write('drawn %d-%d\n' % (survivorCount[0],survivorCount[1]))
            sys.stdout.flush()

            for p, player in enumerate(self.players):
                try:
                    player.new_generation_agents()
                except Exception as e:
                    if in_tournament:
                        self.game_scores[p] = -self.nAgents
                        self.game_messages[p] = str(e)
                        game_play = False
                    else:
                        traceback.print_exc()
                        sys.exit(-1)

            if not game_play:
                return

            if game in save_games:
                #Save a game
                if not os.path.isdir(savePath):
                    os.makedirs(savePath,exist_ok=True)

                now = datetime.now()
                # Month abbreviation, day and year
                saveStr = now.strftime("%b-%d-%Y-%H-%M-%S")
                saveStr += "-%s-vs-%s" % (self.players[0].name,self.players[1].name)
                saveStr += "-game%03d.pickle.gz" % game

                saveFile = os.path.join(savePath, saveStr)

                self.game_saves.append(saveFile)

                with gzip.open(saveFile, 'w') as f:
                    pickle.dump((self.players[0].name,self.players[1].name,self.gridSize,vis_walls,vis_food,vis_agents), f)

            for p in range(2):
                self.game_scores[p] = survivorCount[p]




    # Play visualisation of a saved game
    @staticmethod
    def load(loadGame,visResolution=(720,480), visSpeed='normal'):
        import vis_pygame as vis

        if not os.path.isfile(loadGame):
            print("Error! Saved game file '%s' not found." % loadGame)
            sys.exit(-1)

        # Open the game file and read data
        try:
            with gzip.open(loadGame) as f:
              (player1Name,player2Name,gridSize,vis_walls,vis_food,vis_agents) = pickle.load(f)
        except:
            print("Error! Failed to load %s." % loadGame)

        # Create an instance of visualiser
        v = vis.visualiser(speed=visSpeed, gridSize=gridSize, playerStrings=(player1Name, player2Name),
                       resolution=visResolution)

        # Show visualisation
        for t in range(vis_agents.shape[2]):
            v.show(creature_state=vis_agents[:, :, t], food_array=vis_food[t], wall_array=vis_walls,turn=t)


def main(argv):
    # Load the defaults
    from defaults import game_params

    # Check of arguments from command line
    try:
        opts, args = getopt.getopt(argv, "p:r:v:s:f:l:g:",["players=", "res=", "vis=", "save=", "fast=", "load=", "games="])
    except getopt.GetoptError:
        print("Error! Invalid argument.")
        sys.exit(2)

    # Process command line arguments
    loadGame = None
    for opt, arg in opts:
        if opt in ("-p", "--players"):
            players = arg.split(',')
            if len(players) != 2:
               print("Error! The -p/players= argument must be followed with two comma separated file name (no spaces).")
               sys.exit(-1)

            game_params['player1'] = players[0]
            game_params['player2'] = players[1]

        elif opt in ("-r", "--res"):
            res = arg.split('x')
            if len(res) != 2:
               print("Error! The -r/res= argument must be followed with <width>x<height> specifiction of resolution (no spaces).")
               sys.exit(-1)

            game_params['visResolution'] = (int(res[0]), int(res[1]))
        elif opt in ("-v", "--vis"):
            if arg[0] == '[' and arg[-1] != ']':
                print(
                    "Error! The -v/vis= argument must be followed with [...] giving the list of games to visualise (no spaces).")
                sys.exit(-1)
            game_params['show_games'] = list()
            if len(arg)>2:
                arg = arg[1:-1]
                games = arg.split(',')
                for g in games:
                    game_params['show_games'].append(int(g))

        elif opt in ("-s", "--save"):
            if arg[0] == '[' and arg[-1] != ']':
                print(
                    "Error! The -s/save= argument must be followed with [...] giving the list of games to save (no spaces).")
                sys.exit(-1)
            game_params['save_games'] = list()
            if len(arg)>2:
                arg = arg[1:-1]
                games = arg.split(',')
                for g in games:
                    game_params['save_games'].append(int(g))
        elif opt in ("-f", "--fast"):
            game_params['visSpeed'] = arg

        elif opt in ("-l", "--load"):
            loadGame = arg

        elif opt in ("-g", "--games"):
            game_params['nGames'] = int(arg)

    if game_params['visSpeed'] != 'normal' and game_params['visSpeed'] != 'fast' and game_params['visSpeed'] != 'slow':
        print("Error! Invalid setting '%s' for visualisation speed.  Valid choices are 'slow','normal',fast'" % game_params['visSpeed'])
        sys.exit(-1)

    if loadGame is None:
        # Create a new game and run it

        g = Game(gridSize=game_params['gridSize'],
                 nTurns=game_params['nTurns'],
                 nAgents=game_params['nAgents'],
                 nWalls=game_params['nWalls'],
                 nGames=game_params['nGames'])

        g.run(game_params['player1'],
              game_params['player2'],
              visResolution=game_params['visResolution'],
              visSpeed=game_params['visSpeed'],
              show_games=game_params['show_games'],
              save_games=game_params['save_games'])
    else:
        # Load a previously saved game
        Game.load(loadGame,visResolution=game_params['visResolution'],
               visSpeed=game_params['visSpeed'])



if __name__ == "__main__":
   main(sys.argv[1:])

