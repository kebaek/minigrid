from gym_minigrid.minigrid import *
from gym_minigrid.register import register

# Sparse vs Well Designed IS
class MazeEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

    def __init__(
        self,
        size=7,
        agent_start_pos=(1,5),
        agent_start_dir=0,
        intermediate=False,
        one_intermediate=False
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.intermediate = intermediate
        self.one_intermediate = one_intermediate
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )
        self.actions = MazeEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        self.grid.horz_wall(1, 4, length = 4)
        self.grid.horz_wall(2, 2, length = 4)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, 1)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        #Place cookies
        if self.intermediate:
            self.grid.set(2,1, Ball())
            self.grid.set(2,3, Ball())
            self.grid.set(5,3, Ball())
            self.grid.set(3,5, Ball())

        if self.one_intermediate:
            self.grid.set(3,5, Ball())

        self.mission = "get to the green goal square"

    def step(self, action):
        self.step_count += 1
        info = {'success':False}
        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        if self.step_count >= self.max_steps:
            done = True
        # Rotate left
        elif action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                info['success']=True
                reward = 10
            if fwd_cell != None and fwd_cell.type == 'ball':
                reward = 1
                self.grid.set(*fwd_pos, None)
        else:
            assert False, "unknown action"

        obs = self.gen_obs()

        return obs, reward, done, info
class MazeEnv0(MazeEnv):
    def __init__(self, **kwargs):
        super().__init__(size=7, **kwargs)
class IntermediateMazeEnv0(MazeEnv):
    def __init__(self, **kwargs):
        super().__init__(size=7, intermediate=True)
class OneIntermediateMazeEnv0(MazeEnv):
    def __init__(self, **kwargs):
        super().__init__(size=7, one_intermediate=True)
register(
    id='MiniGrid-Maze-v0',
    entry_point='gym_minigrid.envs:MazeEnv0'
)
register(
    id='MiniGrid-Maze-Intermediate-v0',
    entry_point='gym_minigrid.envs:IntermediateMazeEnv0'
)
register(
    id='MiniGrid-Maze-OneIntermediate-v0',
    entry_point='gym_minigrid.envs:OneIntermediateMazeEnv0'
)

# Sparse vs Well Designed IR
class ThreeDoorsEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        pickup = 3
        toggle = 4

    def __init__(
        self,
        size=9,
        agent_start_pos=(1,7),
        agent_start_dir=3,
        intermediate=False
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )
        self.carrying = []
        self.actions = ThreeDoorsEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        self.grid.horz_wall(2, 1, length = 5)
        self.grid.horz_wall(2, 3, length = 5)
        self.grid.horz_wall(2, 5, length = 5)
        self.grid.horz_wall(2, 7, length = 5)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), 7, 1)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        #Place keys
        self.grid.set(1,2, Key('purple'))
        self.grid.set(4,2, Key('light green'))
        self.grid.set(3,4, Key('blue'))
        self.grid.set(2,6, Key('pink'))

        #Place keys
        self.grid.set(3,2, Door('purple', is_locked=True))
        self.grid.set(6,2, Door('light green', is_locked=True))
        self.grid.set(5,4, Door('blue', is_locked=True))
        self.grid.set(6,6, Door('pink', is_locked=True))

        self.mission = "get to the green goal square"

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False
        info = {'success':False}

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if self.step_count >= self.max_steps:
            done = True
        elif action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                info['success']=True
                done = True
                reward = 10
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                self.carrying.append(fwd_cell.color)
                self.grid.set(*fwd_pos, None)
                reward = 0
        elif action == self.actions.toggle:
            if fwd_cell:
                opened = fwd_cell.toggle(self, fwd_pos)
                if opened:
                    reward = 0
        else:
            assert False, "unknown action"

        obs = self.gen_obs()

        return obs, reward, done, info
class DoorEnv0(ThreeDoorsEnv):
    def __init__(self, **kwargs):
        super().__init__(size=9, **kwargs)
register(
    id='MiniGrid-ThreeDoor-v0',
    entry_point='gym_minigrid.envs:DoorEnv0'
)
# Trade off between Path and Computational Complexity
class FourDoorsEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        pickup = 3
        toggle = 4

    def __init__(
        self,
        size=9,
        agent_start_pos=(1,7),
        agent_start_dir=3,
        intermediate=False
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )
        self.carrying = []
        self.actions = FourDoorsEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.grid.horz_wall(2, 1, length = 5)
        self.grid.horz_wall(2, 3, length = 5)
        self.grid.horz_wall(2, 5, length = 5)
        self.grid.horz_wall(2, 7, length = 5)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), 7, 7)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        #Place keys
        self.grid.set(1,5, Key('orange'))
        self.grid.set(1,2, Key('purple'))
        self.grid.set(4,2, Key('light green'))
        self.grid.set(3,4, Key('blue'))
        self.grid.set(4,6, Key('pink'))

        #Place keys
        self.grid.set(1,3, Door('orange',is_locked=True))
        self.grid.set(3,2, Door('purple',is_locked=True))
        self.grid.set(6,2, Door('light green',is_locked=True))
        self.grid.set(5,4, Door('blue',is_locked=True))
        self.grid.set(6,6, Door('pink',is_locked=True))

        self.mission = "get to the green goal square"

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False
        info = {'success':False}

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        # Rotate left
        if self.step_count >= self.max_steps:
            done = True
        elif action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                info['success']=True
                done = True
                reward = 1000
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                self.carrying.append(fwd_cell.color)
                self.grid.set(*fwd_pos, None)
                reward = 2
        elif action == self.actions.toggle:
            if fwd_cell:
                open = fwd_cell.toggle(self, fwd_pos)
                if open:
                    reward = 2
        elif self.step_count >= self.max_steps:
            done = True
        else:
            assert False, "unknown action"

        obs = self.gen_obs()

        return obs, reward, done, info
class DoorEnv1(FourDoorsEnv):
    def __init__(self, **kwargs):
        super().__init__(size=9, **kwargs)
register(
    id='MiniGrid-FourDoor-v0',
    entry_point='gym_minigrid.envs:DoorEnv1'
)
