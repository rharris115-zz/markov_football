from markov_football.markov_football import *
from markov_football.util import *
from pprint import pprint
from markov_football.name import football_clubs_by_league

if __name__ == '__main__':

    clubs_by_league = football_clubs_by_league()

    lineups_by_league = OrderedDict(
        ((league, {club: create_selection(name=club, players=generate_random_player_population(n=17))
                   for club in clubs})
         for league, clubs in clubs_by_league.items()))

    for league, selections_by_name in lineups_by_league.items():

        names = list(selections_by_name.keys())

        points, wins, draws, losses, goals, conceded_goals = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()

        player_position_history = defaultdict(list)

        for week, fixtures_this_week in enumerate(fixtures(selections_by_name.keys())):
            hold_week(fixtures=fixtures_this_week, selections_by_name=selections_by_name,
                      player_position_history=player_position_history, goals=goals, conceded_goals=conceded_goals,
                      points=points, wins=wins, losses=losses, draws=draws)



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

        for club, selection in selections_by_name.items():
            print(club)
            for player in selection.keys():
                count = Counter(player_position_history[player.name])
                print(player.name, [(position.name, count[position])
                                    for position in Position])
            print()

        break
