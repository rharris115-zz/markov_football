import csv
import os
from typing import Iterable, Tuple
from numpy import random


def names_generator(n: int) -> Iterable[Tuple[str, str]]:
    with open(os.path.join('..', 'names', 'census-dist-male-first.csv')) as file:
        r = csv.reader(file, delimiter=',')
        first_names = [name for name, *rest in r]

    with open(os.path.join('..', 'names', 'census-dist-2500-last.csv')) as file:
        r = csv.reader(file, delimiter=',')
        last_names = [name for name, *rest in r]

    for i in range(n):
        yield (random.choice(first_names), random.choice(last_names))
