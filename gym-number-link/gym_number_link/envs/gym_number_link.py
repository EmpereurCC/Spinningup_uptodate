import numpy as np
import copy
import json
import subprocess
import gin
import inspect, os
from collections import OrderedDict

import gym
from gym import spaces
from gym.utils import seeding
from gym.spaces.discrete import Discrete
from gym.envs.classic_control import rendering

from .colours import colour_dictionary

_author_ = "Alexandre Laterre <a.laterre@instadeep.com>"


@gin.configurable
class NumberLink(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30,
    }

    board_generation = ["random", "generator"]

    def __init__(self,
                 board_size: "(int or tuple) Shape of the board " = 7,
                 num_wires: "(int) number of wires to connect" = 3,
                 board_generation: "(str) method used to generate the wires" = "random",
                 seed: "(int) random seed" = None,
                 numberlink_path: '(str) path to numberlink cpp generator' = None):

        if board_generation not in NumberLink.board_generation:
            ValueError("board_generation does not match any available method, got:", board_generation)
        self.board_generation = board_generation

        self._seed = self.seed(seed=seed)

        if isinstance(board_size, list) or isinstance(board_size, tuple):
            self.board_size = tuple(board_size)
        else:
            self.board_size = (board_size, board_size)
        self.num_wires = num_wires

        self._directions = {0: "E", 1: "S", 2: "W", 3: "N"}
        self._moves = {0: [1, 0], 1: [0, -1], 2: [-1, 0], 3: [0, 1]}
        self.action_space = Discrete(self.num_wires * len(self._directions) * 2)

        self.reward_range = (0.0, 1.0)
        self.observation_space = spaces.Box(
            low=0, high=self.num_wires, shape=(self.board_size[0], self.board_size[1], 2), dtype=np.uint16)

        self._numberlink_path = numberlink_path
        self._compile_generator()
        self.reset()

        self._set_colour_map()
        self.viewer = None

    def seed(self, seed=None):
        if self.board_generation is "generator" and (seed is None or seed < 1):
            raise ValueError('Board with generator must have a specified (int) seed greater than 1.')
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: 
            observation (object): the initial observation.
        """
        self._create_wires()
        self._init_board()
        if self.board_generation == "generator":
            self._seed[0] += 1
        return self._get_observation()

    def _get_observation(self):
        """Return a board representation with the paths and the pins"""
        current_heads = np.zeros(self.board_size, dtype=np.uint16)
        for k, wire in enumerate(self.wires):
            (x0, y0), (x1, y1) = wire.current()
            current_heads[x0, y0] = current_heads[x1, y1] = k + 1
        return np.stack([self.board.astype(np.uint16), current_heads], axis=2)

    def _create_wires(self):
        """Create the wires by placing their pins on the board"""
        if self.board_generation == "random":
            Sx, Tx = zip(*self.np_random.randint(self.board_size[0], size=(self.num_wires, 2)))
            Sy, Ty = zip(*self.np_random.randint(self.board_size[1], size=(self.num_wires, 2)))
            self.wires = []
            for sx, sy, tx, ty in zip(Sx, Sy, Tx, Ty):
                self.wires.append(Wire(np.array([sx, sy]), np.array([tx, ty])))

        if self.board_generation == "generator":
            board = self._generate_wires()
            start_positions = [path[0] for _, path in board.items()]
            end_positions = [path[-1] for _, path in board.items()]

            self.wires = []
            for sp, ep in zip(start_positions, end_positions):
                self.wires.append(Wire(np.array(sp), np.array(ep)))

    def _init_board(self):
        """Initialize the board with the pin values"""
        self.board = np.zeros(self.board_size, dtype=int)
        for k, wire in enumerate(self.wires):
            s, t = wire.pins()
            self.board[s[0], s[1]] = k + 1
            self.board[t[0], t[1]] = k + 1

    def _set_board(self, indices, wire_id=0):
        """Set the value of the board for the given indices to wire_id"""
        for i, j in indices:
            self.board[i, j] = wire_id

    def _display_board(self):
        """Print a string representation of the board"""
        tmp = np.char.mod('%d', np.rot90(self.board)).reshape(-1).tolist()
        tmp = [a.replace('0', ' ') for a in tmp]
        print(np.array(tmp).reshape(self.board_size))

    def _cond_not_connected(self, wire_id) -> bool:
        """True if the wire is not connected already"""
        return not self.wires[wire_id].connected

    def _cond_board_size(self, wire, pin, move) -> bool:
        """True if the wire stays within the boundaries of the board"""
        x, y = self.wires[wire].get_new_position(pin, move)
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]

    def _cond_obstacle(self, wire, pin, move) -> bool:
        """True if the wire goes to an unoccupied space"""
        x, y = self.wires[wire].get_new_position(pin, move)
        return self.board[x, y] == 0 or self.board[x, y] == (wire + 1)

    def _compile_generator(self):
        """Compiles the generator files with Cpp compiler"""
        filename = str(os.path.join(self._numberlink_path, 'gameboard-generator', 'src', 'numberlink.cpp'))
        subprocess.check_call(' '.join(['g++', filename, '-o', 'numberlink']), shell=True)

    def _generate_wires(self):
        """
        Returns dict with paths as list of sublists; the latter representing nodes.
        NOTE: Only generates square boards for now!
        """

        generation_command = [
            './numberlink', '-b',
            str(self.board_size[1]), '-lvl',
            str(self.num_wires), '-s',
            str(self._seed[0]), '-n',
            str(self.board_size[0])
        ]

        path_str = subprocess.Popen(
            generation_command,
            stdout=subprocess.PIPE,
        )
        path_str = path_str.communicate()[0]

        return NumberLink.decode_to_json(path_str)

    @classmethod
    def decode_to_json(cls, text):
        """take string and transform it to json"""
        decoded_utf = text.decode("utf-8")
        return json.loads(decoded_utf)

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        obs, reward, done, info = self._get_observation(), 0.0, False, {}

        # check that the action is within the action space
        if not self.action_space.contains(action):
            return obs, reward, done, info

        wire = action // (2 * len(self._directions))
        action = action % (2 * len(self._directions))
        pin = action // len(self._directions)
        direction = action % len(self._directions)
        move = self._moves[direction]

        # check that the wire is not already connected
        # if so, the move is discarded
        if not self._cond_not_connected(wire):
            return obs, reward, done, info

        # check that the move will stay within the board boundaries
        # if so, the move is discarded
        if not self._cond_board_size(wire, pin, move):
            return obs, reward, done, info

        # chek that the wire will not go to an occupied space
        # if so, the move is discarded
        if not self._cond_obstacle(wire, pin, move):
            return obs, reward, done, info

        A, D, connect = self.wires[wire].move(pin, move)
        self._set_board(A, wire + 1)
        self._set_board(D)

        # receive a reward if the wire get connected
        if connect:
            reward = 1.0 / self.num_wires

        done = np.all([wire.connected for wire in self.wires])
        return self._get_observation(), reward, done, info

    def clone(self) -> object:
        """Clone the object, except the seed"""
        clone = NumberLink(self.board_size, self.num_wires)
        clone.wires = []
        for wire in self.wires:
            clone.wires.append(copy.deepcopy(wire))
        clone.board = np.copy(self.board)
        return clone

    def render(self, mode="human"):
        screen_height = screen_width = 500
        scale_height = screen_height / self.board_size[0]
        scale_width = screen_width / self.board_size[1]
        scales = np.array([scale_height, scale_width])

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

        radius = np.min(scales) / 2
        for k, wire in enumerate(self.wires):

            # display the pins of the wires
            for position in wire.pins():
                t = rendering.Transform(translation=position * scales)
                self.viewer.draw_circle(radius, res=60, color=self.colour_map[k]).add_attr(t)

            # display the paths of each wire
            for path in wire.paths.values():
                previous = None
                for point in path:
                    if previous is not None:
                        self.viewer.draw_line(previous * scales, point * scales, color=self.colour_map[k])
                    previous = point

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def _set_colour_map(self):
        colours = colour_dictionary()
        rgb_tuples = colours.values()
        self.colour_map = OrderedDict(zip(range(self.num_wires), rgb_tuples))

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class Wire(object):

    def __init__(self, pin1, pin2):
        self.paths = {0: [pin1], 1: [pin2]}
        self.connected = np.all(pin1 == pin2)

    def reset(self):
        self.paths[0] = [self.paths[0][0]]
        self.paths[1] = [self.paths[1][0]]
        self.connected = np.all(self.paths[0][0] == self.paths[1][0])

    def get_new_position(self, pin, direction):
        return self.paths[pin][-1] + direction

    def move(self, pin, direction):
        # direction should be an np.array [+/-, +/-]
        assert not self.connected, "The wire is already connected"

        source, target = pin, 1 - pin
        new_position = self.get_new_position(source, direction)
        add_indices, remove_indices = [], []

        # check for a cycle with its own path
        comparison = [np.all(new_position == pt) for pt in self.paths[source]]
        if np.any(comparison):
            index = np.argwhere(comparison)[0][0]
            remove_indices += self.paths[source][index + 1:]
            self.paths[source] = self.paths[source][:index + 1]
        else:
            add_indices += [new_position]
            self.paths[source].append(new_position)

        # check if the wire is connected to any point in the other path
        comparison = [np.all(new_position == pt) for pt in self.paths[target]]
        if np.any(comparison):
            index = np.argwhere(comparison)[0][0]
            remove_indices += self.paths[target][index + 1:]
            self.paths[target] = self.paths[target][:index + 1]
            self.connected = True

        return add_indices, remove_indices, self.connected

    def pins(self):
        return self.paths[0][0], self.paths[1][0]

    def current(self):
        return self.paths[0][-1], self.paths[1][-1]

    def _full_path(self):
        return np.stack(self.paths[0] + self.paths[1][::-1])

    def __str__(self) -> str:
        paths = ["({}, {})".format(i, j) for (i, j) in self.paths[0]]
        if not self.connected:
            paths += ["..."]
        paths += ["({}, {})".format(i, j) for (i, j) in self.paths[1]][::-1]
        return " -- ".join(paths)


if __name__ == '__main__':
    gen_path = '/Users/vcourgeau/go/src/deepswarm/environments/gym-number-link/gym_number_link/envs/'
    gin.parse_config_file('config.gin')
    nl_example = NumberLink()
    # nl_example = NumberLink(board_generation='generator', board_size=10, num_wires=2, numberlink_path=gen_path, seed=42)
    print(nl_example._generate_wires())
    print(nl_example._generate_wires())
    print(nl_example._generate_wires())
    print(nl_example._generate_wires())