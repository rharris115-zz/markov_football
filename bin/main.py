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
        mean_table[position.name] = pd.Series([player_counts_by_position[position]
                                               for player_counts_by_position in
                                               player_counts_by_position_list], index=mean_table.index)

    print(mean_table)


if __name__ == '__main__':

    clubs_by_league = football_clubs_by_league()

    lineups_by_league = OrderedDict(
        ((league, {club: create_selection(name=club, players=generate_random_player_population(n=17))
                   for club in clubs})
         for league, clubs in clubs_by_league.items()))

    for league, selections_by_name in lineups_by_league.items():
        # new_selections = list(optmise_player_positions_in_parrallel(selections=selections_by_name.values(),
        #                                                             team_states=[TeamState.WITH_M],
        #                                                             max_cycles_without_improvement=25))
        # selections_by_name.clear()
        # selections_by_name.update({selection.name for selection in new_selections})

        for week, fixtures_this_week in enumerate(fixtures(selections_by_name.keys())):
            for club_1, club_2 in fixtures_this_week:
                print('Week %d: %s vs. %s' % (week, club_1, club_2))

                selection_1 = selections_by_name[club_1]
                selection_2 = selections_by_name[club_2]

                selection_1, selection_2 = optmise_player_positions_in_parrallel(
                    selections=(selection_1, selection_2),
                    team_states=[TeamState.WITH_M])

                selections_by_name[club_1] = selection_1
                selections_by_name[club_2] = selection_2

                display_league(lineups_by_name={club_1: selection_1,
                                                club_2: selection_2})

                mc = calculate_markov_chain(selection_1=selection_1, selection_2=selection_2)

                score_keeper = Counter()
                s = S(club_1, TeamState.WITH_M)
                for step in range(100):
                    next_s = mc.simulate_next(s)

                    if next_s == S(club_1, TeamState.SCORED):
                        score_keeper.update([club_1])
                        s = S(club_2, TeamState.WITH_M)
                    elif next_s == S(club_2, TeamState.SCORED):
                        score_keeper.update([club_2])
                        s = S(club_1, TeamState.WITH_M)
                    else:
                        s = next_s

                print()
                print('%s: %d\t%s: %d' % (club_1, score_keeper[club_1], club_2, score_keeper[club_2]))
                print()
                print()
            print()
            print()
            print()
