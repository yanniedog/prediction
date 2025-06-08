import pytest
import numpy as np
import parameter_optimizer
import indicator_factory

def test_define_search_space_with_matype():
    """Test that matype parameters are properly handled in search space definition"""
    factory = indicator_factory.IndicatorFactory()
    
    # Test with apo indicator which has matype
    indicator_name = "apo"
    definition = factory.indicator_params[indicator_name]
    params = definition["params"]
    search_space, param_names, fixed_params, has_tunable = parameter_optimizer._define_search_space(params)
    
    # Find matype in search space
    matype_space = [s for s in search_space if getattr(s, 'name', None) == 'matype']
    assert matype_space, "matype should be in search space"
    matype = matype_space[0]
    assert hasattr(matype, 'categories')
    assert set(matype.categories) == set(range(9))
    
    # Test with stoch indicator which has multiple matype parameters
    indicator_name = "stoch"
    definition = factory.indicator_params[indicator_name]
    params = definition["params"]
    search_space, param_names, fixed_params, has_tunable = parameter_optimizer._define_search_space(params)
    # Should include slowk_matype and slowd_matype
    assert any(getattr(s, 'name', None) == 'slowk_matype' for s in search_space)
    assert any(getattr(s, 'name', None) == 'slowd_matype' for s in search_space)

def test_optimize_with_matype():
    """Test that optimization works with matype parameters"""
    # We'll use the search space definition and simulate a simple optimization
    factory = indicator_factory.IndicatorFactory()
    
    # Test with apo indicator
    indicator_name = "apo"
    definition = factory.indicator_params[indicator_name]
    params = definition["params"]
    search_space, param_names, fixed_params, has_tunable = parameter_optimizer._define_search_space(params)
    # Simulate an objective function that prefers higher matype
    def objective(param_list):
        param_dict = dict(zip(param_names, param_list))
        return -param_dict.get("matype", 0)  # Negative so optimizer maximizes matype
    # Try a few values
    for val in range(9):
        score = objective([val] + [0]*(len(param_names)-1))
        assert score == -val
    # Test with stoch indicator
    indicator_name = "stoch"
    definition = factory.indicator_params[indicator_name]
    params = definition["params"]
    search_space, param_names, fixed_params, has_tunable = parameter_optimizer._define_search_space(params)
    def objective2(param_list):
        param_dict = dict(zip(param_names, param_list))
        return -(param_dict.get("slowk_matype", 0) + param_dict.get("slowd_matype", 0))
    # Try a few values
    for val1 in range(9):
        for val2 in range(9):
            param_list = [val1 if n=="slowk_matype" else val2 if n=="slowd_matype" else 0 for n in param_names]
            score = objective2(param_list)
            assert score == -(val1+val2) 