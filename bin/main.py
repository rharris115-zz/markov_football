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

        names = list(selections_by_name.keys())

        points, wins, draws, losses, goals, conceded_goals = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()

        for week, fixtures_this_week in enumerate(fixtures(selections_by_name.keys())):

            for club_1, club_2 in fixtures_this_week:
                print('Week %d: %s vs. %s' % (week, club_1, club_2))

                if not club_1 or not club_2:
                    continue

                selection_1 = selections_by_name[club_1]
                selection_2 = selections_by_name[club_2]

                selection_1, selection_2 = optmise_player_positions_in_parrallel(
                    selections=(selection_1, selection_2),
                    team_states=[TeamState.WITH_M])

                selections_by_name[club_1] = selection_1
                selections_by_name[club_2] = selection_2

                display_league(lineups_by_name={club_1: selection_1,
                                                club_2: selection_2})

                score_keeper = hold_fixture(selection_1=selection_1, selection_2=selection_2)

                goals.update(score_keeper)
                conceded_goals[club_1] += score_keeper[club_2]
                conceded_goals[club_2] += score_keeper[club_1]

                if score_keeper[club_1] > score_keeper[club_2]:
                    points[club_1] += 3
                    wins[club_1] += 1
                    losses[club_2] += 1
                elif score_keeper[club_2] > score_keeper[club_1]:
                    points[club_2] += 3
                    wins[club_2] += 1
                    losses[club_1] += 1
                else:
                    points[club_1] += 1
                    points[club_2] += 1
                    draws[club_1] += 1
                    draws[club_2] += 1

                print()
                print('%s: %d\t%s: %d' % (club_1, score_keeper[club_1], club_2, score_keeper[club_2]))
                print()
                print()

            data = OrderedDict([('p', [points[name] for name in names]),
                                ('w', [wins[name] for name in names]),
                                ('d', [draws[name] for name in names]),
                                ('l', [losses[name] for name in names]),
                                ('g', [goals[name] for name in names]),
                                ('c', [conceded_goals[name] for name in names]),
                                ('gd', [goals[name] - conceded_goals[name] for name in names])])
            table = pd.DataFrame(data=data, index=names)
            table.sort_values(['p', 'gd', 'g'], ascending=[False, False, False], inplace=True)

            print('Table after week %d.' % week)
            print(table)
            print()
