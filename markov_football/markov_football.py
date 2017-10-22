from typing import Tuple, Dict, List, Iterable
from .markov import MarkovChain, Tx
from .name_generator import names_generator
from enum import Enum, auto
from collections import namedtuple, UserList, UserDict, defaultdict
from numpy import random
import math

goal_keeper_correction = 1.0


class TeamState(Enum):
    GK = auto()
    D = auto()
    M = auto()
    F = auto()
    SCORED = auto()


S = namedtuple('S', ['team', 'team_state'])


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


def generate_player_population(n: int = 1) -> Iterable[Player]:
    ng = names_generator(n=n)
    for i in range(n):
        native_abilities = Abilities(
            {ability: value for ability, value in zip(Ability, random.uniform(low=0.0, high=1.0,
                                                                              size=len(Ability)))})
        player = Player(name=next(ng), age=16, abilities=native_abilities)
        yield player


class TeamLineup(object):
    def __init__(self, name: str, goal_keeper: Player, defenders: List[Player], midfielders: List[Player],
                 forwards: List[Player]):

        if not goal_keeper:
            raise ValueError('Need a goal keeper!')

        n_defenders = len(defenders)
        n_midfielders = len(midfielders)
        n_forwards = len(forwards)

        if n_defenders + n_midfielders + n_forwards > 10:
            raise ValueError('Too many players! n_defenders=%d, n_midfielders=%d, n_forwards=%d' % (
                n_defenders, n_midfielders, n_forwards))

        self.name = name
        self.goal_keeper = goal_keeper
        self.defenders = tuple(defenders)
        self.midfielders = tuple(midfielders)
        self.forwards = tuple(forwards)

    def goal_keeper_ability(self, ability: Ability) -> float:
        return goal_keeper_correction * self.goal_keeper.abilities[ability]

    def total_in_defense(self, ability: Ability) -> float:
        return sum(map(lambda p: p.abilities[ability], self.defenders))

    def total_in_midfield(self, ability: Ability) -> float:
        return sum(map(lambda p: p.abilities[ability], self.midfielders))

    def total_forward(self, ability: Ability) -> float:
        return sum(map(lambda p: p.abilities[ability], self.forwards))


def create_lineup_with_typical_abilities(name: str, typical: float = 0.5) -> TeamLineup:
    ng = names_generator(n=11)

    def player_gen():
        native_abilities = Abilities(
            {ability: typical for ability in Ability})
        return Player(name=next(ng), age=16, abilities=native_abilities)

    return TeamLineup(name=name,
                      goal_keeper=player_gen(),
                      defenders=[player_gen() for i in range(4)],
                      midfielders=[player_gen() for i in range(4)],
                      forwards=[player_gen() for i in range(2)])


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _calculate_team_probs(lineup: TeamLineup, other_lineup: TeamLineup) -> List[Tx]:
    name = lineup.name
    other_name = other_lineup.name

    gk_passing = lineup.goal_keeper_ability(Ability.PASSING)

    d_passing = lineup.total_in_defense(Ability.PASSING)
    m_passing = lineup.total_in_midfield(Ability.PASSING)
    f_passing = lineup.total_forward(Ability.PASSING)

    of_intercepting = other_lineup.total_forward(Ability.INTERCEPTION)
    om_intercepting = other_lineup.total_in_midfield(Ability.INTERCEPTION)
    od_intercepting = other_lineup.total_in_defense(Ability.INTERCEPTION)

    d_dribbling = lineup.total_in_defense(Ability.DRIBBLING)
    m_dribbling = lineup.total_in_midfield(Ability.DRIBBLING)
    f_dribbling = lineup.total_forward(Ability.DRIBBLING)

    of_tackling = other_lineup.total_forward(Ability.TACKLING)
    om_tackling = other_lineup.total_in_midfield(Ability.TACKLING)
    od_tackling = other_lineup.total_in_defense(Ability.TACKLING)
    ogk_tackling = other_lineup.goal_keeper_ability(Ability.TACKLING)

    f_shooting = lineup.total_forward(Ability.SHOOTING)

    od_blocking = other_lineup.total_in_defense(Ability.BLOCKING)
    ogk_blocking = other_lineup.goal_keeper_ability(Ability.BLOCKING)

    p_gk_d = logistic(gk_passing - of_intercepting)
    p_gk_m = logistic(gk_passing - om_intercepting)

    p_d_d = logistic(d_passing + d_dribbling - of_tackling)
    p_d_m = logistic(d_passing + d_dribbling - of_tackling - om_intercepting)

    p_m_m = logistic(m_passing + m_dribbling - om_tackling)
    p_m_f = logistic(m_passing + m_dribbling - om_tackling - od_intercepting)

    p_f_f = logistic(f_passing + f_dribbling - od_tackling)
    p_f_sc = logistic(f_shooting + f_dribbling - od_tackling - ogk_tackling - od_blocking - ogk_blocking)

    return [
        Tx(S(name, TeamState.GK), S(name, TeamState.D), p_gk_d),
        Tx(S(name, TeamState.GK), S(other_name, TeamState.F), 1.0 - p_gk_d),
        Tx(S(name, TeamState.GK), S(name, TeamState.M), p_gk_m),
        Tx(S(name, TeamState.GK), S(other_name, TeamState.M), 1.0 - p_gk_m),

        Tx(S(name, TeamState.D), S(name, TeamState.D), p_d_d),
        Tx(S(name, TeamState.D), S(other_name, TeamState.F), 1.0 - p_d_d),
        Tx(S(name, TeamState.D), S(name, TeamState.M), p_d_m),
        Tx(S(name, TeamState.D), S(other_name, TeamState.M), 1.0 - p_d_m),

        Tx(S(name, TeamState.M), S(name, TeamState.M), p_m_m),
        Tx(S(name, TeamState.M), S(other_name, TeamState.M), 1.0 - p_m_m),
        Tx(S(name, TeamState.M), S(name, TeamState.F), p_m_f),
        Tx(S(name, TeamState.M), S(other_name, TeamState.D), 1.0 - p_m_f),

        Tx(S(name, TeamState.F), S(name, TeamState.F), p_f_f),
        Tx(S(name, TeamState.F), S(other_name, TeamState.D), 1.0 - p_f_f),
        Tx(S(name, TeamState.F), S(name, TeamState.SCORED), p_f_sc),
        Tx(S(name, TeamState.F), S(other_name, TeamState.GK), 1.0 - p_f_sc)
    ]


def calculate_markov_chain(lineup1: TeamLineup, lineup2: TeamLineup) -> MarkovChain:
    return MarkovChain(_calculate_team_probs(lineup=lineup1, other_lineup=lineup2) +
                       _calculate_team_probs(lineup=lineup2, other_lineup=lineup1))


def _next_goal_probs(mc: MarkovChain, lineup: TeamLineup,
                     reference_lineup: TeamLineup,
                     team_states: Iterable[TeamState]) -> Dict[S, float]:
    next_goal_probs = defaultdict()
    for team_state in team_states:
        e_absorbing_state_probabilities = mc.calculate_outcome_given_state(S(lineup.name, team_state))
        r_absorbing_state_probabilities = mc.calculate_outcome_given_state(S(reference_lineup.name, team_state))


def evaluate_next_goal_probs(player: Player,
                             existing_lineup: TeamLineup,
                             reference_lineup: TeamLineup,
                             team_states: Iterable[TeamState] = (TeamState.M)) -> TeamLineup:
    original_mc = calculate_markov_chain(lineup1=existing_lineup, lineup2=reference_lineup)

    {team_state: (
        original_mc.calculate_outcome_given_state(S(existing_lineup.name, team_state)),
        original_mc.calculate_outcome_given_state(S(reference_lineup.name, team_state)))
        for team_state in team_states}

    for team_state in team_states:
        e_absorbing_state_probabilities = original_mc.calculate_outcome_given_state(S(existing_lineup.name, team_state))
        r_absorbing_state_probabilities = original_mc.calculate_outcome_given_state(
            S(reference_lineup.name, team_state))
