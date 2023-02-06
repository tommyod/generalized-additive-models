#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 21:39:38 2023

@author: tommy
"""

import numpy as np
import pytest

from generalized_additive_models.links import LINKS


class TestLink:
    @pytest.mark.parametrize("link", LINKS.values())
    def test_that_links_are_inverses_link_mu(self, link):
        rng = np.random.default_rng(42)
        argument = rng.random(10_000)

        link_instance = link()
        forward = link_instance.link(argument)
        back_again = link_instance.inverse_link(forward)

        assert np.allclose(back_again, argument)

    @pytest.mark.parametrize("link", LINKS.values())
    def test_that_links_are_inverses_mu_link(self, link):
        rng = np.random.default_rng(42)
        argument = rng.random(10_000)

        link_instance = link()
        backward = link_instance.inverse_link(argument)
        forward_again = link_instance.link(backward)

        assert np.allclose(forward_again, argument)

    @pytest.mark.parametrize("link", LINKS.values())
    def test_that_links_derivatives_are_close_to_finite_differences(self, link):
        rng = np.random.default_rng(42)
        argument = 0.01 + rng.random(100) * 0.98
        epsilon = np.ones_like(argument) * 1e-9  # 8, 9, 10 seems to work

        # Derivative from equation vs. finite difference approximation to the derivative
        f_x_deriv = link().gradient(argument)
        f_x_finite_diff = (link().link(argument + epsilon) - link().link(argument)) / epsilon

        # Atleast 90% must be close. This test is not exact. Depends on random
        # numbers and numerics...
        assert np.allclose(f_x_deriv, f_x_finite_diff)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys"])
