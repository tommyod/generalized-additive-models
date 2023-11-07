#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 21:39:38 2023

@author: tommy
"""

import inspect

import numpy as np
import pytest

from generalized_additive_models.links import LINKS


class TestLink:
    @pytest.mark.parametrize("link", LINKS.values())
    def test_that_links_are_inverses_link_mu(self, link):
        rng = np.random.default_rng(42)
        argument = rng.random(10_000)

        # If the signature contains low, create it
        signature_parameters = set(inspect.signature(link).parameters.keys())
        if all(param in signature_parameters for param in ("low", "high")):
            low = argument - rng.random(len(argument))
            high = argument + rng.random(len(argument))
            link_instance = link(low=low, high=high)

        elif "low" in signature_parameters:
            low = argument - rng.random(len(argument))
            link_instance = link(low=low)

        elif "high" in signature_parameters:
            high = argument + rng.random(len(argument))
            link_instance = link(high=high)
        else:
            link_instance = link()

        # Compute forward and back again
        forward = link_instance.link(argument)
        back_again = link_instance.inverse_link(forward)

        assert np.allclose(back_again, argument)

    @pytest.mark.parametrize("link", LINKS.values())
    def test_that_links_are_inverses_mu_link(self, link):
        rng = np.random.default_rng(43)
        argument = rng.random(10_000)

        # If the signature contains low, create it
        signature_parameters = set(inspect.signature(link).parameters.keys())
        if all(param in signature_parameters for param in ("low", "high")):
            low = argument - rng.random(len(argument))
            high = argument + rng.random(len(argument))
            link_instance = link(low=low, high=high)

        elif "low" in signature_parameters:
            low = argument - rng.random(len(argument))
            link_instance = link(low=low)

        elif "high" in signature_parameters:
            high = argument + rng.random(len(argument))
            link_instance = link(high=high)
        else:
            link_instance = link()

        # Compute backward, then forward again
        backward = link_instance.inverse_link(argument)
        forward_again = link_instance.link(backward)

        assert np.allclose(forward_again, argument)

    @pytest.mark.parametrize("link", LINKS.values())
    def test_that_links_derivatives_are_close_to_finite_differences(self, link):
        rng = np.random.default_rng(42)
        argument = 0.01 + rng.random(1000) * 0.98
        epsilon = np.ones_like(argument) * 1e-8  # from 5 to 11 works, use 8

        # Derivative from equation vs. finite difference approximation to the derivative
        f_x_deriv = link().derivative(argument)
        f_x_finite_diff = (link().link(argument + epsilon) - link().link(argument - epsilon)) / (2 * epsilon)

        # At least 90% must be close. This test is not exact. Depends on random
        # numbers and numerics...
        assert np.allclose(f_x_deriv, f_x_finite_diff)

    @pytest.mark.parametrize("link", LINKS.values())
    def test_that_links_second_derivatives_are_close_to_finite_differences(self, link):
        rng = np.random.default_rng(41)
        argument = 0.01 + rng.random(1000) * 0.98
        epsilon = np.ones_like(argument) * 1e-6  # from 5 to 8 works, use 6

        l = link()

        # Derivative from equation vs. finite difference approximation to the derivative
        f_x_deriv2 = l.second_derivative(argument)
        dfx_finite_diff = (l.derivative(argument + epsilon) - l.derivative(argument - epsilon)) / (2 * epsilon)

        # At least 90% must be close. This test is not exact. Depends on random
        # numbers and numerics...
        assert np.allclose(f_x_deriv2, dfx_finite_diff)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys"])
