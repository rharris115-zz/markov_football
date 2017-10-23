from typing import Tuple, Dict, List, Iterable, NamedTuple, Generator
from .markov import MarkovChain, Tx
from .name import NamesGenerator
from enum import Enum, auto
from collections import UserDict, defaultdict, OrderedDict
import numpy as np
import math
from pprint import pprint
import pandas as pd

goal_keeper_correction = 3.0


class Position(Enum):
    GK = auto()
    D = auto()
    M = auto()
    F = auto()


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
    TACKLING = auto()
    INTERCEPTION = auto()
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


class TeamLineup(object):
    pass


class TeamLineup(dict):
    def __init__(self, name: str, players: Dict[Player, Position] = {}):

        if len(players) > 11:
            raise ValueError('Too many players! len(players)=%d' % len(players))

        if len(list(filter(lambda item: item[1] is Position.GK, players.items()))) > 1:
            raise ValueError('Can only have zero or one Goal Keepers.')

        super().__init__(players.items())

        self.name = name

    def __repr__(self):
        return self.__class__.__name__ + '(' + self.name + ': ' + super().__repr__() + ')'

    def total_ability(self, ability: Ability, position: Position) -> float:
        return sum(map(lambda item: item[0].abilities[ability]
        if item[1] is position
        else 0.0, self.items())) * (goal_keeper_correction if position is Position.GK else 1.0)

    def with_addition(self, player: Player, position: Position) -> TeamLineup:
        players = OrderedDict(list(self.items()) + [(player, position)])
        return TeamLineup(name=self.name, players=players)

    def with_substitution(self, player: Player, substitute: Player) -> TeamLineup:
        if player not in self:
            raise ValueError("Cannot find player to be substituted. player=%s" % player)
        position = self[player]
        players = dict(self)
        del players[player]
        players[substitute] = position
        return TeamLineup(name=self.name, players=players)

    def with_player_positions(self, player_positions: List[Tuple[Player, Position]]) -> TeamLineup:
        players = dict(self)
        for player, position in player_positions:
            players[player] = position
        return TeamLineup(name=self.name, players=players)


def generate_random_player_population(n: int = 1) -> Iterable[Player]:
    ng = NamesGenerator.names(n=n)
    for i in range(n):
        abilities = Abilities(
            {ability: value for ability, value in zip(Ability, np.random.uniform(low=0.0, high=1.0,
                                                                                 size=len(Ability)))})
        player = Player(name=next(ng), age=16, abilities=abilities)
        yield player


def generate_typical_player_population(n: int = 1, typical: float = 0.5) -> Iterable[Player]:
    ng = NamesGenerator.names(n=n)
    for i in range(n):
        abilities = Abilities(
            {ability: typical for ability in Ability})
        player = Player(name=next(ng), age=16, abilities=abilities)
        yield player


def create_lineup(name: str, players: Iterable[Player]) -> TeamLineup:
    return TeamLineup(name=name,
                      players=OrderedDict([(next(players), Position.GK)] +
                                          [(next(players), Position.D) for i in range(4)] +
                                          [(next(players), Position.M) for i in range(4)] +
                                          [(next(players), Position.F) for i in range(2)]))


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _calculate_team_probs(lineup: TeamLineup, other_lineup: TeamLineup) -> List[Tx]:
    name = lineup.name
    other_name = other_lineup.name

    gk_passing = lineup.total_ability(Ability.PASSING, Position.GK)

    d_passing = lineup.total_ability(Ability.PASSING, Position.D)
    m_passing = lineup.total_ability(Ability.PASSING, Position.M)
    f_passing = lineup.total_ability(Ability.PASSING, Position.F)

    of_intercepting = other_lineup.total_ability(Ability.INTERCEPTION, Position.F)
    om_intercepting = other_lineup.total_ability(Ability.INTERCEPTION, Position.M)
    od_intercepting = other_lineup.total_ability(Ability.INTERCEPTION, Position.D)

    d_dribbling = lineup.total_ability(Ability.DRIBBLING, Position.D)
    m_dribbling = lineup.total_ability(Ability.DRIBBLING, Position.M)
    f_dribbling = lineup.total_ability(Ability.DRIBBLING, Position.F)

    of_tackling = other_lineup.total_ability(Ability.TACKLING, Position.F)
    om_tackling = other_lineup.total_ability(Ability.TACKLING, Position.M)
    od_tackling = other_lineup.total_ability(Ability.TACKLING, Position.D)
    ogk_tackling = other_lineup.total_ability(Ability.TACKLING, Position.GK)

    m_shooting = lineup.total_ability(Ability.SHOOTING, Position.M)
    f_shooting = lineup.total_ability(Ability.SHOOTING, Position.F)

    om_blocking = other_lineup.total_ability(Ability.BLOCKING, Position.M)
    od_blocking = other_lineup.total_ability(Ability.BLOCKING, Position.D)
    ogk_blocking = other_lineup.total_ability(Ability.BLOCKING, Position.GK)

    p_gk_d = logistic(gk_passing - of_intercepting)
    p_gk_m = logistic(gk_passing - om_intercepting)
    p_gk_f = logistic(gk_passing - od_intercepting)

    p_d_d = logistic(d_passing + d_dribbling - of_tackling)
    p_d_m = logistic(d_passing + d_dribbling - of_tackling - om_intercepting)

    p_m_m = logistic(m_passing + m_dribbling - om_tackling)
    p_m_f = logistic(m_passing + m_dribbling - om_tackling - od_intercepting)
    p_m_sc = logistic(m_shooting + m_dribbling - om_tackling - om_blocking - od_tackling - od_blocking - ogk_blocking)

    p_f_f = logistic(f_passing + f_dribbling - od_tackling)
    p_f_sc = logistic(f_shooting + f_dribbling - od_tackling - od_blocking - ogk_blocking)

    return [
        # GK pass to D
        Tx(S(name, TeamState.WITH_GK), S(name, TeamState.WITH_D), p_gk_d),
        Tx(S(name, TeamState.WITH_GK), S(other_name, TeamState.WITH_F), 1.0 - p_gk_d),

        # GK pass to M
        Tx(S(name, TeamState.WITH_GK), S(name, TeamState.WITH_M), p_gk_m),
        Tx(S(name, TeamState.WITH_GK), S(other_name, TeamState.WITH_M), 1.0 - p_gk_m),

        # GK pass to F
        Tx(S(name, TeamState.WITH_GK), S(name, TeamState.WITH_F), p_gk_f),
        Tx(S(name, TeamState.WITH_GK), S(other_name, TeamState.WITH_D), 1.0 - p_gk_f),

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
        Tx(S(name, TeamState.WITH_F), S(name, TeamState.WITH_F), p_f_f),
        Tx(S(name, TeamState.WITH_F), S(other_name, TeamState.WITH_D), 1.0 - p_f_f),

        # F shoots
        Tx(S(name, TeamState.WITH_F), S(name, TeamState.SCORED), p_f_sc),
        Tx(S(name, TeamState.WITH_F), S(other_name, TeamState.WITH_GK), 1.0 - p_f_sc)
    ]


def calculate_markov_chain(lineup1: TeamLineup, lineup2: TeamLineup) -> MarkovChain:
    return MarkovChain(_calculate_team_probs(lineup=lineup1, other_lineup=lineup2) +
                       _calculate_team_probs(lineup=lineup2, other_lineup=lineup1))


def next_goal_probs(mc: MarkovChain,
                    team_states: Iterable[TeamState]) -> Dict[S, float]:
    names = [name for name, ts in mc.absorbing_states]
    return mc.calculate_mean_outcome_given_states(
        (S(name, team_state)
         for team_state in team_states
         for name in names))


def optimise_player_positions(
        original_lineup: TeamLineup,
        reference_lineups: List[TeamLineup],
        team_states: Iterable[TeamState],
        max_trials_without_improvement: int = 1000) -> TeamLineup:
    best_mean_next_goal_prob = sum(evaluate_lineup(lineup=original_lineup, reference_lineups=reference_lineups,
                                                   team_states=team_states)) / len(reference_lineups)
    best_lineup = original_lineup
    trials_without_imrovement = 0
    while trials_without_imrovement < max_trials_without_improvement:
        new_mean_next_goal_prob, new_lineup = _experiment_with_positioning(lineup=best_lineup,
                                                                           reference_lineups=reference_lineups,
                                                                           team_states=team_states)
        if not new_lineup:
            continue
        if new_mean_next_goal_prob > best_mean_next_goal_prob:
            best_mean_next_goal_prob = new_mean_next_goal_prob
            best_lineup = new_lineup
            trials_without_imrovement = 0
        trials_without_imrovement += 1

    return best_lineup


def _experiment_with_positioning(lineup: TeamLineup,
                                 reference_lineups: List[TeamLineup],
                                 team_states: Iterable[TeamState]) -> Tuple[float, TeamLineup]:
    if np.random.choice(a=[True, False]):
        player = np.random.choice(a=list(lineup.keys()))
        old_position = lineup[player]
        new_position = np.random.choice(a=[pos for pos in Position if pos is not old_position])
        try:
            new_lineup = lineup.with_player_positions(player_positions=[(player, new_position)])
        except:
            return (0, None)
    else:
        player1, player2 = np.random.choice(a=list(lineup.keys()),
                                            size=2,
                                            replace=False)
        position1, position2 = lineup[player1], lineup[player2]
        if position1 is position2:
            return (0, None)
        try:
            new_lineup = lineup.with_player_positions(
                player_positions=[(player1, position2), (player2, position1)])
        except:
            return (0, None)

    new_next_goal_prob = sum(evaluate_lineup(lineup=new_lineup,
                                             reference_lineups=reference_lineups,
                                             team_states=team_states)) / len(reference_lineups)
    return new_next_goal_prob, new_lineup


def evaluate_lineup(
        lineup: TeamLineup,
        reference_lineups: List[TeamLineup],
        team_states: Iterable[TeamState]) -> Iterable[float]:
    for reference_lineup in reference_lineups:
        next_goal_prob = 0.5 if reference_lineup is lineup else \
            next_goal_probs(mc=calculate_markov_chain(lineup1=lineup,
                                                      lineup2=reference_lineup),
                            team_states=team_states)[S(lineup.name, TeamState.SCORED)]
        yield next_goal_prob


def create_next_goal_matrix(lineups: List[TeamLineup], team_states: Iterable[TeamState]) -> pd.DataFrame:
    names = [lineup.name for lineup in lineups]
    n = len(lineups)
    A = np.zeros(shape=(n, n))
    for row_index, lineup in enumerate(lineups):
        A[row_index, :] = list(evaluate_lineup(lineup=lineup,
                                               reference_lineups=lineups,
                                               team_states=team_states))
    frame = pd.DataFrame(data=pd.DataFrame(A, index=names, columns=names))
    return frame
