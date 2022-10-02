class Session(object):
    pass

class Node(object):
    pass


def new_session(endpoint, user_credential) -> Session:
    pass


def get_nodes(sess: Session, filter=None) -> [Node]:
    pass


def fed_avg(
    sess: Session,
    initial_model,
    nodes=None,       # what nodes are included
    num_rounds=1,
    aggregator=None,  # algorithm ID or aggregator object
    learner=None,     # algorithm ID or a learner object
):
    pass


def cyclic(
    sess: Session,
    initial_model,
    nodes=None,
    num_rounds=1,
    learner=None,     # algorithm ID or a Learner object
):
    pass


def solve(
    sess: Session,
    data,
    nodes=None,
    solver=None,  # Solver ID or Solver object
) -> [object]:
    pass
