import pytest

from llm.router import (
    RouterContractError,
    _parse_routing_response,
    _validate_routing_contract,
)


def test_analysis_sweep_cannot_route_to_multipoint_optimization():
    prompt = (
        "Analyze a tapered wing. Sweep alpha from -4 to 16 deg at Mach 0.45 "
        "and Mach 0.55. Plot CL vs alpha, L/D vs alpha, and drag polar."
    )
    data = _parse_routing_response(
        '<routing>{"blueprints":["aero_multipoint.py"],"is_vague":false}</routing>'
    )

    with pytest.raises(RouterContractError):
        _validate_routing_contract(data, prompt)


def test_optimization_request_can_route_to_multipoint():
    prompt = (
        "Optimize drag across Mach 0.6 and Mach 0.78 with twist_cp as a design "
        "variable and CL constraints at both points."
    )
    data = _parse_routing_response(
        '<routing>{"blueprints":["aero_multipoint.py"],"is_vague":false}</routing>'
    )

    assert _validate_routing_contract(data, prompt)["blueprints"] == [
        "aero_multipoint.py"
    ]


def test_router_rejects_multiple_blueprints():
    with pytest.raises(RouterContractError):
        _parse_routing_response(
            '<routing>{"blueprints":["aero_analysis.py","aero_opt.py"]}</routing>'
        )
