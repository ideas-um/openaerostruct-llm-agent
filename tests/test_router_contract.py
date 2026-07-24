import pytest

from llm.router import (
    RouterContractError,
    _parse_routing_response,
    _validate_routing_contract,
)


def test_analysis_sweep_keeps_router_agent_blueprint_decision():
    prompt = (
        "Analyze a tapered wing. Sweep alpha from -4 to 16 deg at Mach 0.45 "
        "and Mach 0.55. Plot CL vs alpha, L/D vs alpha, and drag polar."
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


def test_router_does_not_replace_invalid_agent_choice_with_default():
    with pytest.raises(RouterContractError):
        _parse_routing_response(
            '<routing>{"blueprints":["unsupported_blueprint.py"]}</routing>'
        )


def test_router_keeps_only_supported_parameter_fields():
    prompt = "Minimize drag using tube thickness."
    data = _parse_routing_response(
        """
        <routing>
        {
          "blueprints": ["aerostruct_tube.py"],
          "is_vague": false,
          "parameters": {
            "objective": "minimize drag",
            "design_variables": [
              {
                "name": "thickness_cp",
                "bounds": {"values": [0.005, 0.1], "unit": "m"}
              }
            ],
            "loads": [
              {
                "direction": "upward",
                "magnitude": {"value": 40000, "unit": "N"},
                "distribution": "spanwise nodes",
                "domain": "whole wing"
              }
            ],
            "settings": {"viscous": true, "wave": false},
            "unsupported_guess": "do not pass this"
          }
        }
        </routing>
        """
    )

    validated = _validate_routing_contract(data, prompt)

    assert validated["parameters"] == {
        "objective": "minimize drag",
        "design_variables": [
            {
                "name": "thickness_cp",
                "bounds": {"values": [0.005, 0.1], "unit": "m"},
            }
        ],
        "loads": [
            {
                "direction": "upward",
                "magnitude": {"value": 40000, "unit": "N"},
                "distribution": "spanwise nodes",
                "domain": "whole wing",
            }
        ],
        "settings": {"viscous": True, "wave": False},
    }
