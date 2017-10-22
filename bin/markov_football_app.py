from markov_football.markov_football import *
from pprint import pprint
import numpy as np

if __name__ == '__main__':

    typical_lineup = create_lineup(name='typical', players=generate_typical_player_population(n=11, typical=0.5))

    home_lineup = create_lineup(name='home', players=generate_random_player_population(n=11))

    away_lineup = create_lineup(name='away', players=generate_random_player_population(n=11))

    assigned_lineup = create_lineup(name='assigned', players=generate_random_player_population(n=11))
    assigned_lineup = optimise_player_positions(original_lineup=assigned_lineup, reference_lineup=typical_lineup)

    tmc = calculate_markov_chain(lineup1=home_lineup, lineup2=typical_lineup)
    mc = calculate_markov_chain(lineup1=home_lineup, lineup2=away_lineup)
    smc = calculate_markov_chain(lineup1=assigned_lineup, lineup2=typical_lineup)

    initial_state = S('home', TeamState.WITH_M)
    s = initial_state

    print(next_goal_probs(mc=tmc))
    print(next_goal_probs(mc=mc))
    print(next_goal_probs(mc=smc))

    exit()

    home_score, away_score = 0, 0

    for step in range(1000):
        next_s = mc.simulate_next(s)

        if next_s == S('home', TeamState.SCORED):
            home_score += 1
            s = S('away', TeamState.WITH_M)
        elif next_s == S('away', TeamState.SCORED):
            away_score += 1
            s = S('home', TeamState.WITH_M)
        else:
            s = next_s

        print(step, home_score, away_score, s)
