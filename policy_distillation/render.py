import pygame
from pygame import gfxdraw
import numpy as np


def render_cartpole(state, env_params):
    screen_width = 600
    screen_height = 400
    length = 0.5
    x_threshold = 2.4

    pygame.init()
    screen = pygame.Surface((screen_width, screen_height))

    world_width = x_threshold * 2
    scale = screen_width / world_width
    polewidth = 10.0
    polelen = scale * (2 * length)
    cartwidth = 50.0
    cartheight = 30.0
    tau = env_params.tau

    if state is None:
        return None

    x = state

    surf = pygame.Surface((screen_width, screen_height))
    surf.fill((255, 255, 255))

    for draw_mode in [0, 1]:

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0

        cartx = (x[0] + tau * x[1] * draw_mode) * scale + screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(surf, cart_coords, (0, 0, 0, 250 - draw_mode * 150))
        gfxdraw.filled_polygon(surf, cart_coords, (0, 0, 0, 250 - draw_mode * 150))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-(x[2] + tau * x[3] * draw_mode))
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(surf, pole_coords, (202, 152, 101, 250 - draw_mode * 150))
        gfxdraw.filled_polygon(surf, pole_coords, (202, 152, 101, 250 - draw_mode * 150))

        gfxdraw.aacircle(
            surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203, 200 - draw_mode * 120),
        )
        gfxdraw.filled_circle(
            surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203, 200 - draw_mode * 120),
        )

        gfxdraw.hline(surf, 0, screen_width, carty, (0, 0, 0))

    surf = pygame.transform.flip(surf, False, True)
    screen.blit(surf, (0, 0))

    return np.transpose(
        np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
    )