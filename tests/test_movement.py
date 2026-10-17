from primary.data.board import ADJACENCY, NUM_REGIONS, Region
from primary.engine.movement import distance, reachable


def test_one_step_is_adjacency():  # rule 11
    for region in Region:
        assert set(reachable(region, 1)) == set(ADJACENCY[region])


def test_new_england_within_two_steps():  # rule 21
    assert set(reachable(Region.NEW_ENGLAND, 2)) == {
        Region.NORTH_EAST,
        Region.EARLY_MIDWEST,
        Region.APPALACHIA,
    }


def test_new_england_within_three_steps():  # rule 48
    assert set(reachable(Region.NEW_ENGLAND, 3)) == {
        Region.NORTH_EAST,
        Region.EARLY_MIDWEST,
        Region.APPALACHIA,
        Region.MIDWEST,
        Region.GREAT_LAKES,
        Region.SOUTH,
    }


def test_reachable_excludes_start():
    assert Region.MIDWEST not in reachable(Region.MIDWEST, 3)


def test_distance_west_to_new_england():
    assert distance(Region.WEST, Region.NEW_ENGLAND) == 5
    assert distance(Region.WEST, Region.WEST) == 0


def test_map_is_connected():
    for region in Region:
        assert len(reachable(region, NUM_REGIONS)) == NUM_REGIONS - 1
