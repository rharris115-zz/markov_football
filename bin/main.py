from markov_football.markov_football import *
from pprint import pprint
from markov_football.name import football_clubs_by_league

if __name__ == '__main__':

    clubs_by_league = football_clubs_by_league()

    lineups_by_league = OrderedDict()
    for league, clubs in clubs_by_league.items():
        lineups_by_league[league] = [create_lineup(name=club, players=generate_random_player_population(n=11))
                                     for club in clubs]

    for league, lineups in lineups_by_league.items():
        print(league)
        print(create_next_goal_matrix(lineups, team_states=[TeamState.WITH_M]))
        new_lineups = [optimise_player_positions(original_lineup=lineup,
                                                 reference_lineups=[other_lineup
                                                                    for other_lineup in lineups
                                                                    if other_lineup is not lineup],
                                                 team_states=[TeamState.WITH_M],
                                                 max_trials_without_improvement=10)
                       for lineup in lineups]
        lineups[:] = new_lineups




        # for step in range(1000):
        #     next_s = mc.simulate_next(s)
        #
        #     if next_s == S('home', TeamState.SCORED):
        #         home_score += 1
        #         s = S('away', TeamState.WITH_M)
        #     elif next_s == S('away', TeamState.SCORED):
        #         away_score += 1
        #         s = S('home', TeamState.WITH_M)
        #     else:
        #         s = next_s
        #
        #     print(step, home_score, away_score, s)
