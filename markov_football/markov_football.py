from typing import Tuple, Dict, List, Iterable, NamedTuple, Generator
from .markov import MarkovChain, Tx
from .name import NamesGenerator
from enum import Enum, auto
from collections import UserDict, defaultdict, OrderedDict, Counter
import numpy as np
import math
import pandas as pd
import logging

logger = logging.getLogger()
logger.setLevel(logging.WARN)
logger.addHandler(logging.StreamHandler())

goal_keeper_correction = 3.0


class Position(Enum):
    B = auto()
    GK = auto()
    D = auto()
    M = auto()
    F = auto()


outfield_positions = frozenset((Position.D, Position.M, Position.F))


class TeamState(Enum):
    WITH_GK = auto()
    WITH_D = auto()
    WITH_M = auto()
    WITH_F = auto()
    SCORED = auto()


class S(NamedTuple):
    team: str
    team_state: TeamState


class Ability(Enum):
    BLOCKING = auto()
    BALL_WINNING = auto()
    SHOOTING = auto()
    DRIBBLING = auto()
    PASSING = auto()


class Abilities(UserDict):
    def __init__(self, abilities: Dict[Ability, float] = {}):
        abilities.update({ability: 0.0 for ability in Ability if ability not in abilities})
        super().__init__(abilities)


class Player(object):
    def __init__(self, name: Tuple[str, str], age: int, abilities: Abilities):
        self.name = name
        self.age = age
        self.abilities = abilities

    def __repr__(self):
        return '{name=%r, age=%d, abilities=%r}' % (
            self.name, self.age, self.abilities)


class Selection(object):
    pass


class Selection(dict):
    def __init__(self, name: str, players: Iterable[Tuple[Player, Position]] = ()):

        if not name:
            raise ValueError('Need a name.')

        position_counter = Counter(position for player, position in players)

        outfield_count = sum(position_counter[position]
                             for position in outfield_positions)

        if outfield_count > 10:
            raise ValueError('Cannot have more than %d in positions %s.' % (10, ','.join(position.name
                                                                                         for position in
                                                                                         outfield_positions)))

        if position_counter[Position.GK] > 1:
            raise ValueError('Cannot have more than %d in position %s.' % (1, Position.GK.name))

        super().__init__(players)

        self.name = name

    def __repr__(self):
        return self.__class__.__name__ + '(' + self.name + ': ' + super().__repr__() + ')'

    def total_ability(self, ability: Ability, position: Position) -> float:
        players_at_position = {player: position for player, play_position in self.items() if play_position is position}
        return math.sqrt(sum(map(lambda item: item[0].abilities[ability],
                                 players_at_position.items())) * (
                             goal_keeper_correction if position is Position.GK else 1.0))

    def formation(self) -> Dict[Position, List[Player]]:
        f = defaultdict(list)
        for player, position in self.items():
            f[position].append(player)
        return OrderedDict(((position, f[position]) for position in Position))

    def with_addition(self, player: Player, position: Position) -> Selection:
        players = list(self.items() + [(player, position)])
        return Selection(name=self.name, players=players)

    def with_substitution(self, player: Player, substitute: Player) -> Selection:
        if player not in self:
            raise ValueError("Cannot find player to be substituted. player=%s" % player)
        position = self[player]
        players = dict(self)
        del players[player]
        players[substitute] = position
        return Selection(name=self.name, players=players.items())

    def with_player_positions(self, player_positions: List[Tuple[Player, Position]]) -> Selection:
        players = dict(self)
        for player, position in player_positions:
            if player not in players:
                raise ValueError("Cannot find player to play in position %s. player=%s" % (player, position))

            players[player] = position
        return Selection(name=self.name, players=players.items())


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x / 4))


def _calculate_team_probs(selection: Selection, other_selection: Selection) -> List[Tx]:
    name = selection.name
    other_name = other_selection.name

    gk_passing = selection.total_ability(Ability.PASSING, Position.GK)
    d_passing = selection.total_ability(Ability.PASSING, Position.D)
    m_passing = selection.total_ability(Ability.PASSING, Position.M)
    f_passing = selection.total_ability(Ability.PASSING, Position.F)

    f_ball_winning = selection.total_ability(Ability.BALL_WINNING, Position.F)
    m_ball_winning = selection.total_ability(Ability.BALL_WINNING, Position.M)
    d_ball_winning = selection.total_ability(Ability.BALL_WINNING, Position.D)

    of_ball_winning = other_selection.total_ability(Ability.BALL_WINNING, Position.F)
    om_ball_winning = other_selection.total_ability(Ability.BALL_WINNING, Position.M)
    od_ball_winning = other_selection.total_ability(Ability.BALL_WINNING, Position.D)

    d_dribbling = selection.total_ability(Ability.DRIBBLING, Position.D)
    m_dribbling = selection.total_ability(Ability.DRIBBLING, Position.M)
    f_dribbling = selection.total_ability(Ability.DRIBBLING, Position.F)

    m_shooting = selection.total_ability(Ability.SHOOTING, Position.M)
    f_shooting = selection.total_ability(Ability.SHOOTING, Position.F)

    om_blocking = other_selection.total_ability(Ability.BLOCKING, Position.M)
    od_blocking = other_selection.total_ability(Ability.BLOCKING, Position.D)
    ogk_blocking = other_selection.total_ability(Ability.BLOCKING, Position.GK)

    p_gk_d = logistic(gk_passing + d_ball_winning - of_ball_winning)
    p_gk_m = logistic(gk_passing + m_ball_winning - om_ball_winning)
    p_gk_f = logistic(gk_passing + f_ball_winning - od_ball_winning)

    p_d_d = logistic(d_passing + d_dribbling + d_ball_winning - of_ball_winning)
    p_d_m = logistic(d_passing + d_dribbling + m_ball_winning - of_ball_winning)

    p_m_m = logistic(m_passing + m_dribbling + m_ball_winning - om_ball_winning)
    p_m_f = logistic(m_passing + m_dribbling + f_ball_winning - od_ball_winning)
    p_m_sc = logistic(m_shooting + m_dribbling - om_blocking - od_blocking - ogk_blocking)

    p_f_f = logistic(f_passing + f_dribbling + f_ball_winning + f_ball_winning - od_ball_winning)
    p_f_sc = logistic(f_shooting + f_dribbling - od_blocking - ogk_blocking)

    return [
        # GK pass to D
        Tx(S(name, TeamState.WITH_GK), S(name, TeamState.WITH_D), p_gk_d),
        Tx(S(name, TeamState.WITH_GK), S(other_name, TeamState.WITH_F), 1.0 - p_gk_d),

        # GK pass to M
        # Tx(S(name, TeamState.WITH_GK), S(name, TeamState.WITH_M), p_gk_m),
        # Tx(S(name, TeamState.WITH_GK), S(other_name, TeamState.WITH_M), 1.0 - p_gk_m),

        # GK pass to F
        # Tx(S(name, TeamState.WITH_GK), S(name, TeamState.WITH_F), p_gk_f),
        # Tx(S(name, TeamState.WITH_GK), S(other_name, TeamState.WITH_D), 1.0 - p_gk_f),

        # D pass to D
        # Tx(S(name, TeamState.WITH_D), S(name, TeamState.WITH_D), p_d_d),
        # Tx(S(name, TeamState.WITH_D), S(other_name, TeamState.WITH_F), 1.0 - p_d_d),

        # D pass to M
        Tx(S(name, TeamState.WITH_D), S(name, TeamState.WITH_M), p_d_m),
        Tx(S(name, TeamState.WITH_D), S(other_name, TeamState.WITH_M), 1.0 - p_d_m),

        # M pass to M
        # Tx(S(name, TeamState.WITH_M), S(name, TeamState.WITH_M), p_m_m),
        # Tx(S(name, TeamState.WITH_M), S(other_name, TeamState.WITH_M), 1.0 - p_m_m),

        # M pass to F
        Tx(S(name, TeamState.WITH_M), S(name, TeamState.WITH_F), p_m_f),
        Tx(S(name, TeamState.WITH_M), S(other_name, TeamState.WITH_D), 1.0 - p_m_f),

        # M shoots
        Tx(S(name, TeamState.WITH_M), S(name, TeamState.SCORED), p_m_sc),
        Tx(S(name, TeamState.WITH_M), S(other_name, TeamState.WITH_GK), 1.0 - p_m_sc),

        # F pass to F
        # Tx(S(name, TeamState.WITH_F), S(name, TeamState.WITH_F), p_f_f),
        # Tx(S(name, TeamState.WITH_F), S(other_name, TeamState.WITH_D), 1.0 - p_f_f),

        # F shoots
        Tx(S(name, TeamState.WITH_F), S(name, TeamState.SCORED), p_f_sc),
        Tx(S(name, TeamState.WITH_F), S(other_name, TeamState.WITH_GK), 1.0 - p_f_sc)
    ]


def calculate_markov_chain(selection_1: Selection, selection_2: Selection) -> MarkovChain:
    return MarkovChain(_calculate_team_probs(selection=selection_1, other_selection=selection_2) +
                       _calculate_team_probs(selection=selection_2, other_selection=selection_1))


def next_goal_probs(mc: MarkovChain,
                    team_states: Iterable[TeamState]) -> Dict[S, float]:
    names = [name for name, ts in mc.absorbing_states]
    return mc.calculate_mean_outcome_given_states(states=(S(name, team_state)
                                                          for team_state in team_states
                                                          for name in names))
