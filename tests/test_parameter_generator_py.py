import parameter_generator

def test_generate_param_grid_basic():
    params = {"a": [1, 2], "b": [3, 4]}
    grid = list(parameter_generator.generate_param_grid(params))
    assert {"a": 1, "b": 3} in grid
    assert {"a": 2, "b": 4} in grid
    assert len(grid) == 4

def test_generate_param_grid_empty():
    params = {}
    grid = list(parameter_generator.generate_param_grid(params))
    assert grid == [{}] 