from typing import Tuple, Dict, List, Iterable, NamedTuple, Generator
from .markov import MarkovChain, Tx
from .name_generator import NamesGenerator
from enum import Enum, auto
from collections import UserDict, defaultdict, OrderedDict
from numpy import random
import math
import pprint

goal_keeper_correction = 2.0


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
        return '{name=%r, age=%d, native_abilities=%r}' % (
            self.name, self.age, self.abilities)


class TeamLineup(object):
    pass


class TeamLineup(object):
    def __init__(self, name: str, players: Dict[Player, Position] = {}):

        if len(players) > 11:
            raise ValueError('Too many players! len(players)=%d' % len(players))

        if len(list(filter(lambda item: item[1] is Position.GK, players.items()))) > 1:
            raise ValueError('Can only have zero or one Goal Keepers.')

        self.name = name
        self.players = OrderedDict(players)

    def total_ability(self, ability: Ability, position: Position) -> float:
        return sum(map(lambda item: item[0].abilities[ability]
        if item[1] is position
        else 0.0, self.players.items()))

    def with_addition(self, player: Player, position: Position) -> TeamLineup:
        players = OrderedDict(list(self.players.items()) + [(player, position)])
        return TeamLineup(name=self.name, players=players)

    def with_substitution(self, player: Player, substitute: Player) -> TeamLineup:
        if player not in self.players:
            raise ValueError("Cannot find player to be substituted. player=%s" % player)
        position = self.players[player]
        players = OrderedDict(self.players)
        del players[player]
        players[substitute] = position
        return TeamLineup(name=self.name, players=players)


def generate_random_player_population(n: int = 1) -> Iterable[Player]:
    ng = NamesGenerator.names(n=n)
    for i in range(n):
        abilities = Abilities(
            {ability: value for ability, value in zip(Ability, random.uniform(low=0.0, high=1.0,
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

    f_shooting = lineup.total_ability(Ability.SHOOTING, Position.F)

    od_blocking = other_lineup.total_ability(Ability.BLOCKING, Position.D)
    ogk_blocking = other_lineup.total_ability(Ability.BLOCKING, Position.GK)

    p_gk_d = logistic(gk_passing - of_intercepting)
    p_gk_m = logistic(gk_passing - om_intercepting)

    p_d_d = logistic(d_passing + d_dribbling - of_tackling)
    p_d_m = logistic(d_passing + d_dribbling - of_tackling - om_intercepting)

    p_m_m = logistic(m_passing + m_dribbling - om_tackling)
    p_m_f = logistic(m_passing + m_dribbling - om_tackling - od_intercepting)

    p_f_f = logistic(f_passing + f_dribbling - od_tackling)
    p_f_sc = logistic(f_shooting + f_dribbling - od_tackling - ogk_tackling - od_blocking - ogk_blocking)

    return [
        Tx(S(name, TeamState.WITH_GK), S(name, TeamState.WITH_D), p_gk_d),
        Tx(S(name, TeamState.WITH_GK), S(other_name, TeamState.WITH_F), 1.0 - p_gk_d),
        Tx(S(name, TeamState.WITH_GK), S(name, TeamState.WITH_M), p_gk_m),
        Tx(S(name, TeamState.WITH_GK), S(other_name, TeamState.WITH_M), 1.0 - p_gk_m),

        Tx(S(name, TeamState.WITH_D), S(name, TeamState.WITH_D), p_d_d),
        Tx(S(name, TeamState.WITH_D), S(other_name, TeamState.WITH_F), 1.0 - p_d_d),
        Tx(S(name, TeamState.WITH_D), S(name, TeamState.WITH_M), p_d_m),
        Tx(S(name, TeamState.WITH_D), S(other_name, TeamState.WITH_M), 1.0 - p_d_m),

        Tx(S(name, TeamState.WITH_M), S(name, TeamState.WITH_M), p_m_m),
        Tx(S(name, TeamState.WITH_M), S(other_name, TeamState.WITH_M), 1.0 - p_m_m),
        Tx(S(name, TeamState.WITH_M), S(name, TeamState.WITH_F), p_m_f),
        Tx(S(name, TeamState.WITH_M), S(other_name, TeamState.WITH_D), 1.0 - p_m_f),

        Tx(S(name, TeamState.WITH_F), S(name, TeamState.WITH_F), p_f_f),
        Tx(S(name, TeamState.WITH_F), S(other_name, TeamState.WITH_D), 1.0 - p_f_f),
        Tx(S(name, TeamState.WITH_F), S(name, TeamState.SCORED), p_f_sc),
        Tx(S(name, TeamState.WITH_F), S(other_name, TeamState.WITH_GK), 1.0 - p_f_sc)
    ]


def calculate_markov_chain(lineup1: TeamLineup, lineup2: TeamLineup) -> MarkovChain:
    return MarkovChain(_calculate_team_probs(lineup=lineup1, other_lineup=lineup2) +
                       _calculate_team_probs(lineup=lineup2, other_lineup=lineup1))


def next_goal_probs(mc: MarkovChain,
                    team_states: Iterable[TeamState] = [TeamState.WITH_M]) -> Dict[S, float]:
    names = [name for name, ts in mc.absorbing_states]
    return mc.calculate_mean_outcome_given_states(
        (S(name, team_state)
         for team_state in team_states
         for name in names))


def build_team(name: str, players: Iterable[Player], reference_lineup: TeamLineup):
    lineup = TeamLineup(name=name)
    for player in players:
        lineup = propose_player_addition(player=player, original_lineup=lineup, reference_lineup=reference_lineup)
        if len(lineup.players) == 11:
            break
    pprint.pprint(lineup.players)
    return lineup


def propose_player_addition(player: Player,
                            original_lineup: TeamLineup,
                            reference_lineup: TeamLineup,
                            team_states: Iterable[TeamState] = [TeamState.WITH_M]) -> TeamLineup:
    original_mc = calculate_markov_chain(lineup1=original_lineup, lineup2=reference_lineup)

    scoring_state = S(original_lineup.name, TeamState.SCORED)

    original_next_goal_prob = next_goal_probs(mc=original_mc, team_states=team_states)[scoring_state]

    best_lineup, best_next_goal_prob = original_lineup, original_next_goal_prob

    if len(original_lineup.players) < 11:

        for position in Position:
            try:
                new_lineup = original_lineup.with_addition(player, position)
            except:
                continue
            new_mc = calculate_markov_chain(lineup1=new_lineup, lineup2=reference_lineup)
            new_next_goal_prob = next_goal_probs(mc=new_mc, team_states=team_states)[scoring_state]

            if new_next_goal_prob > best_next_goal_prob:
                best_next_goal_prob = new_next_goal_prob
                best_lineup = new_lineup

    return best_lineup
