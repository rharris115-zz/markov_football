from .markov_football import *
from collections import deque
from itertools import islice


def generate_random_player_population(n: int = 1) -> Iterable[Player]:
    ng = NamesGenerator.names(n=n)
    multiplier = np.random.uniform(0.0, 2.0)
    for i in range(n):
        abilities = Abilities(
            {ability: multiplier * value for ability, value in zip(Ability, np.random.uniform(low=0.0, high=1.0,
                                                                                              size=len(Ability)))})
        player = Player(name=next(ng), age=16, abilities=abilities)
        yield player


def generate_typical_player_population(n: int = 1, typical: float = 0.5) -> Iterable[Player]:
    ng = NamesGenerator.names(n=n)
    for i in range(n):
        abilities = Abilities(
            {ability: typical for ability in Ability})
        player = Player(name=next(ng), age=16, abilities=abilities)
        yield player


def create_selection(name: str, players: Iterable[Player]) -> Selection:
    return Selection(name=name,
                     players=[(next(players), Position.B) for i in range(6)] +
                             [(next(players), Position.GK)] +
                             [(next(players), Position.D) for i in range(4)] +
                             [(next(players), Position.M) for i in range(4)] +
                             [(next(players), Position.F) for i in range(2)])


def optmise_player_positions_in_parrallel(
        selections: Iterable[Selection],
        team_states: Iterable[TeamState],
        max_cycles_without_improvement: int = 100) -> Iterable[Selection]:
    local_selections_by_name = {selection.name: selection for selection in selections}
    names = [selection.name for selection in selections]

    cycles_without_improvement = 0

    while cycles_without_improvement < max_cycles_without_improvement:

        for name in names:
            selection = local_selections_by_name[name]

            next_goal_p = sum(
                evaluate_selection(selection=selection,
                                   reference_selections=local_selections_by_name.values(),
                                   team_states=team_states)) / len(local_selections_by_name)

            trial_next_goal_p, trial_selection, description = _experiment_with_positioning(selection=selection,
                                                                                           reference_selections=local_selections_by_name.values(),
                                                                                           team_states=team_states)

            if not trial_selection:
                continue
            elif trial_next_goal_p > next_goal_p:
                local_selections_by_name[name] = trial_selection
                logger.info('Change by %s: %s' % (name, description))
                cycles_without_improvement = 0
        cycles_without_improvement += 1
    for name in names:
        yield local_selections_by_name[name]


def _experiment_with_positioning(selection: Selection,
                                 reference_selections: Iterable[Selection],
                                 team_states: Iterable[TeamState]) -> Tuple[float, Selection, str]:
    if np.random.choice(a=[True, False]):
        player = np.random.choice(a=list(selection.keys()))
        old_position = selection[player]
        new_position = np.random.choice(a=[pos for pos in Position if pos is not old_position])
        description = 'Move %s from %s to %s.' % (str(player.name), old_position.name, new_position.name)
        try:
            new_selection = selection.with_player_positions(player_positions=[(player, new_position)])
        except:
            return (0, None, description)
    else:
        player1, player2 = np.random.choice(a=list(selection.keys()),
                                            size=2,
                                            replace=False)
        position1, position2 = selection[player1], selection[player2]
        description = 'Swap %s in %s for %s in %s.' % (
            str(player1.name), position1.name, str(player2.name), position2.name)
        if position1 is position2:
            return (0, None, description)
        try:
            new_selection = selection.with_player_positions(
                player_positions=[(player1, position2), (player2, position1)])
        except:
            return (0, None, description)

    next_goal_probs = list(evaluate_selection(selection=new_selection,
                                              reference_selections=reference_selections,
                                              team_states=team_states))

    new_next_goal_prob = sum(next_goal_probs) / len(next_goal_probs)
    return new_next_goal_prob, new_selection, description


def evaluate_selection(
        selection: Selection,
        reference_selections: Iterable[Selection],
        team_states: Iterable[TeamState]) -> Iterable[float]:
    for reference_selection in reference_selections:
        if reference_selection.name is selection.name:
            yield 0.5
            continue

        ngps = next_goal_probs(mc=calculate_markov_chain(selection_1=selection,
                                                         selection_2=reference_selection),
                               team_states=team_states)
        next_goal_prob = ngps[S(selection.name, TeamState.SCORED)]
        yield next_goal_prob


def create_next_goal_matrix(selections: List[Selection], team_states: Iterable[TeamState]) -> pd.DataFrame:
    names = [selection.name for selection in selections]
    n = len(names)
    A = np.zeros(shape=(n, n))

    mean_probability_other_selection = list()

    for row_index, selection in enumerate(selections):
        probs = list(evaluate_selection(selection=selection,
                                        reference_selections=selections,
                                        team_states=team_states))
        A[row_index, :] = probs

        mean_probability_other_selection.append(
            sum([prob for col_index, prob in enumerate(probs) if col_index != row_index]) / (n - 1))

    frame = pd.DataFrame(data=pd.DataFrame(A, index=names, columns=names))
    frame['mean'] = pd.Series(mean_probability_other_selection, index=frame.index)

    frame.sort_values(['mean'], inplace=True, ascending=False)

    cols = frame.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    frame = frame[cols]

    return frame


def fixtures(teams: Iterable[str]) -> Iterable[str]:
    teams = list(teams)
    if len(teams) % 2:
        teams.append(None)

    ln = len(teams) // 2
    dq1, dq2 = deque(islice(teams, None, ln)), deque(islice(teams, ln, None))
    for _ in range(len(teams) - 1):
        yield list(zip(dq1, dq2))  # list(zip.. python3
        #  pop off first deque's left element to
        # "fix one of the competitors in the first column"
        start = dq1.popleft()
        # rotate the others clockwise one position
        # by swapping elements
        dq1.appendleft(dq2.popleft())
        dq2.append(dq1.pop())
        # reattach first competitor
        dq1.appendleft(start)
