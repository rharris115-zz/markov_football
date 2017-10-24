from markov_football.markov_football import *
from pprint import pprint
from markov_football.name import football_clubs_by_league


def display_league(lineups_by_name: Dict[str, List[TeamLineup]]):
    table = create_next_goal_matrix(lineups_by_name.values(), team_states=[TeamState.WITH_M])
    mean_table = table.loc[:, 'mean']
    print(mean_table)
    top_lineup_name = mean_table.index[0]
    print(top_lineup_name)
    for position, players in lineups_by_name[top_lineup_name].formation().items():
        print('%s: %s' % (position, ','.join(map(str, map(lambda p: p.name, players)))))
    print()


if __name__ == '__main__':

    clubs_by_league = football_clubs_by_league()

    lineups_by_league = OrderedDict(
        ((league, {club: create_lineup(name=club, players=generate_random_player_population(n=11))
                   for club in clubs})
         for league, clubs in clubs_by_league.items()))

    for league, lineups_by_name in lineups_by_league.items():
        print('%s: Initial player allocation' % league)
        display_league(lineups_by_name=lineups_by_name)

        for optimisation in range(1, 100):
            new_lineups = {name: optimise_player_positions(original_lineup=lineup,
                                                           reference_lineups=lineups_by_name.values(),
                                                           team_states=[TeamState.WITH_M],
                                                           max_trials_without_improvement=3)
                           for name, lineup in lineups_by_name.items()}
            lineups_by_name.clear()
            lineups_by_name.update(new_lineups)

            display_league(lineups_by_name=lineups_by_name)

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
