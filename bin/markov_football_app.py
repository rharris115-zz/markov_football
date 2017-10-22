from markov_football.markov_football import generate_player_population, TeamLineup, TeamState, calculate_markov_chain, \
    S, create_lineup_with_typical_abilities
from pprint import pprint
import numpy as np

if __name__ == '__main__':

    gpp = generate_player_population(22)

    typical_lineup = create_lineup_with_typical_abilities(name='typical')

    home_lineup = TeamLineup(name='home',
                             goal_keeper=next(gpp),
                             defenders=[next(gpp) for i in range(4)],
                             midfielders=[next(gpp) for i in range(4)],
                             forwards=[next(gpp) for i in range(2)])
    away_lineup = TeamLineup(name='away',
                             goal_keeper=next(generate_player_population()),
                             defenders=[next(gpp) for i in range(4)],
                             midfielders=[next(gpp) for i in range(4)],
                             forwards=[next(gpp) for i in range(2)])

    tmc = calculate_markov_chain(lineup1=home_lineup, lineup2=typical_lineup)

    mc = calculate_markov_chain(lineup1=home_lineup, lineup2=away_lineup)

    initial_state = S('home', TeamState.M)
    s = initial_state

    print(tmc.calculate_outcome_given_state(S('home', TeamState.M)),
          tmc.calculate_outcome_given_state(S('typical', TeamState.M)))

    home_score, away_score = 0, 0

    for step in range(1000):
        next_s = mc.simulate_next(s)

        if next_s == S('home', TeamState.SCORED):
            home_score += 1
            s = S('away', TeamState.M)
        elif next_s == S('away', TeamState.SCORED):
            away_score += 1
            s = S('home', TeamState.M)
        else:
            s = next_s

        print(step, home_score, away_score, s)
