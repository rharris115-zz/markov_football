from markov_football.markov_football import *
from markov_football.util import *
from pprint import pprint
from markov_football.name import football_clubs_by_league


def display_league(lineups_by_name: Dict[str, List[Selection]]):
    table = create_next_goal_matrix(lineups_by_name.values(), team_states=[TeamState.WITH_M])
    mean_table = table.loc[:, ['mean']]

    player_counts_by_position_list = [
        {position: len(players)
         for position, players in lineups_by_name[lineup_name].formation().items()}
        for lineup_name in mean_table.index
    ]

    for position in Position:
        # d = {str(position): [player_counts_by_position[position]
        #                      for player_counts_by_position in
        #                      player_counts_by_position_list]}

        mean_table[position.name] = pd.Series([player_counts_by_position[position]
                                               for player_counts_by_position in
                                               player_counts_by_position_list], index=mean_table.index)

    print(mean_table)
    print()


if __name__ == '__main__':

    clubs_by_league = football_clubs_by_league()

    lineups_by_league = OrderedDict(
        ((league, {club: create_selection(name=club, players=generate_random_player_population(n=17))
                   for club in clubs})
         for league, clubs in clubs_by_league.items()))

    for league, lineups_by_name in lineups_by_league.items():
        print('%s: Initial player allocation' % league)
        display_league(lineups_by_name=lineups_by_name)

        for optimisation in range(1, 10):
            new_lineups_by_name = optmise_player_positions_in_parrallel(selections_by_name=lineups_by_name,
                                                                        team_states=[TeamState.WITH_M],
                                                                        max_cycles_without_improvement=10)

            lineups_by_name.clear()
            lineups_by_name.update(new_lineups_by_name)

            print('%s: optimasation %d' % (league, optimisation))
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
