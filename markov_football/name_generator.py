import csv
import os
from typing import Iterable, Tuple
from numpy import random


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
