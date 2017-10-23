import csv
import os
from typing import Iterable, Tuple, List, Dict
from numpy import random
from collections import OrderedDict


class NamesGenerator(object):
    _first_names = None
    _last_names = None

    @staticmethod
    def names(n: int) -> Iterable[Tuple[str, str]]:
        if not NamesGenerator._first_names:
            with open(os.path.join('..', 'names', 'census-dist-male-first.csv')) as file:
                r = csv.reader(file, delimiter=',')
                NamesGenerator._first_names = [name for name, *rest in r]

        if not NamesGenerator._last_names:
            with open(os.path.join('..', 'names', 'census-dist-2500-last.csv')) as file:
                r = csv.reader(file, delimiter=',')
                NamesGenerator._last_names = [name for name, *rest in r]

        for i in range(n):
            yield (random.choice(NamesGenerator._first_names), random.choice(NamesGenerator._last_names))


def football_clubs_by_league() -> Dict[str, List[str]]:
    clubs_by_league = OrderedDict()
    with open(os.path.join('..', 'names', 'england_clubs.csv')) as file:
        r = csv.reader(file, delimiter=',')
        for club_name, league, *rest in r:
            clubs = clubs_by_league.get(league)
            if not clubs:
                clubs = list()
                clubs_by_league[league] = clubs
            clubs.append(club_name)
    return clubs_by_league
