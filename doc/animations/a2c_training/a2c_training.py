"""
A2C training on Lunar Lander: 128 parallel envs, rollout buffer, returns,
one big batch, losses, training loop, learning progress.
Manim Community Edition script (tested with manim 0.22).

Put lunar_lander.png (your screenshot) next to this file.
Without it, a vector drawing is used instead.

Optional: paste your training log into TRAINING_LOG (part 8) to plot your real curve.

This file contains ONE Manim scene (A2CTraining) built from 8 parts,
so you always get one video file.

Full video, 1080p 30 fps:
    python a2c_training.py
    # or
    manim -pqh --fps 30 a2c_training.py

Quick 480p preview of selected parts (numbers = part order 1..8):
    A2C_PARTS="6,7" manim -pql a2c_training.py
"""

from manim import *
import numpy as np
import os


# =====================================================================
# Style (same palette as the first video)
# =====================================================================
LOGIT_COLOR  = "#4169E1"     # royalblue
PROB_COLOR   = "#E03C31"     # red
VALUE_COLOR  = "#FF8C00"     # darkorange
REWARD_COLOR = "#FF4D4D"
STATE_COLOR  = "#2EC4B6"     # teal
DONE_COLOR   = YELLOW
TEXT_DIM     = GREY_B

LANDER_BODY  = "#7B68EE"     # purple lander, as in the env
FLAG_COLOR   = "#E6D900"
FLAME_COLOR  = "#FF9F1C"

# one color per discrete action: 0 nothing, 1 left engine, 2 main engine, 3 right engine
ACTION_COLORS = ["#8A8F98", "#F4A261", "#E63946", "#2A9D8F"]

N_ENVS    = 128
N_STATE   = 8
N_ACTIONS = 4
GRID_ROWS, GRID_COLS = 8, 16

HERE = os.path.dirname(os.path.abspath(__file__))
SCREENSHOT = os.path.join(HERE, "lunar_lander.png")


# =====================================================================
# Helpers
# =====================================================================
def section_header(text, color=WHITE):
    return Text(text, font_size=36, color=color).to_edge(UP, buff=0.4)


def make_lander(height=1.0, flame=None):
    """
    Simple vector lander: purple body + two legs.
    flame: None, "main", "left" or "right" -> draws an engine flame.
    """
    body = Polygon([-0.30, 0.25, 0], [0.30, 0.25, 0], [0.38, -0.20, 0], [-0.38, -0.20, 0],
                   fill_color=LANDER_BODY, fill_opacity=1, stroke_width=0)
    leg_l = Line([-0.32, -0.15, 0], [-0.52, -0.50, 0], color=LANDER_BODY, stroke_width=6)
    leg_r = Line([0.32, -0.15, 0], [0.52, -0.50, 0], color=LANDER_BODY, stroke_width=6)
    parts = [body, leg_l, leg_r]
    if flame == "main":
        parts.append(Polygon([-0.14, -0.22, 0], [0.14, -0.22, 0], [0, -0.75, 0],
                             fill_color=FLAME_COLOR, fill_opacity=1, stroke_width=0))
    elif flame == "left":     # left orientation engine: jet on the left side
        parts.append(Polygon([-0.38, 0.12, 0], [-0.38, -0.05, 0], [-0.78, 0.03, 0],
                             fill_color=FLAME_COLOR, fill_opacity=1, stroke_width=0))
    elif flame == "right":
        parts.append(Polygon([0.38, 0.12, 0], [0.38, -0.05, 0], [0.78, 0.03, 0],
                             fill_color=FLAME_COLOR, fill_opacity=1, stroke_width=0))
    g = VGroup(*parts)
    g.scale_to_fit_height(height * (1.25 if flame == "main" else 1.0))
    return g


def make_tile(w, h):
    """One environment: black sky, white ground, two small flags."""
    sky = Rectangle(width=w, height=h, fill_color=BLACK, fill_opacity=1,
                    stroke_color=GREY_D, stroke_width=1.5)
    gy = -h / 2 + h * 0.18
    ground = Polygon([-w / 2, -h / 2, 0], [-w / 2, gy + h * 0.04, 0], [-w * 0.25, gy + h * 0.07, 0],
                     [-w * 0.12, gy, 0], [w * 0.12, gy, 0], [w * 0.3, gy + h * 0.08, 0],
                     [w / 2, gy + h * 0.02, 0], [w / 2, -h / 2, 0],
                     fill_color=WHITE, fill_opacity=1, stroke_width=0)
    flags = VGroup(*[Line([x, gy, 0], [x, gy + h * 0.16, 0], color=FLAG_COLOR, stroke_width=1.5)
                     for x in (-w * 0.12, w * 0.12)])
    return VGroup(sky, ground, flags)


def heatmap(values, color, height, width, lo=None, hi=None, diverging=False, neg_color=None):
    """
    Tensor as a pixel image (one pixel per entry), sharp nearest-neighbour scaling.
    values: (rows, cols). diverging=True -> negative = blue, positive = color.
    """
    v = np.asarray(values, dtype=float)
    lo = v.min() if lo is None else lo
    hi = v.max() if hi is None else hi
    c = np.array(color_to_rgb(color))
    base = np.array([0.06, 0.06, 0.08])
    if diverging:
        m = max(abs(lo), abs(hi), 1e-6)
        t = np.clip(v / m, -1, 1)[..., None]
        if neg_color is None:
            neg_color = LOGIT_COLOR if color != LOGIT_COLOR else "#9AD1FF"
        neg = np.array(color_to_rgb(neg_color))
        rgb = np.where(t >= 0, base + t * (c - base), base + (-t) * (neg - base))
    else:
        t = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)[..., None]
        rgb = base + (0.15 + 0.85 * t) * (c - base)
    img = ImageMobject((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    img.stretch_to_fit_height(height)
    img.stretch_to_fit_width(width)
    return img


def categorical_strip(cats, colors, height, width):
    """Column of discrete values (e.g. actions), one colored pixel per entry."""
    rgb = np.array([color_to_rgb(colors[int(c)]) for c in cats])[:, None, :]
    img = ImageMobject((rgb * 255).astype(np.uint8))
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    img.stretch_to_fit_height(height)
    img.stretch_to_fit_width(width)
    return img


def load_screenshot(height):
    """Your screenshot (window title bar cropped) or a vector fallback."""
    if os.path.exists(SCREENSHOT):
        from PIL import Image
        im = Image.open(SCREENSHOT).convert("RGB")
        w, h = im.size
        im = im.crop((0, 30, w, h))                    # drop the window title bar
        mob = ImageMobject(np.array(im))
        mob.set_height(height)
        mob.px_size = im.size                          # (600, 400)
        return mob
    # fallback: a big vector tile
    tile = make_tile(height * 1.5, height)
    lander = make_lander(height * 0.12).move_to(tile.get_center() + UP * height * 0.33)
    g = Group(tile, lander)
    g.px_size = None
    return g


def fit_right_column(block, max_width=7.0, center=RIGHT * 2.75 + DOWN * 0.2):
    """Place a text block in the right column next to the screenshot, never off-screen."""
    if block.width > max_width:
        block.scale_to_fit_width(max_width)
    return block.move_to(center)


def px_to_point(mob, px, py):
    """Pixel coordinate in the (cropped) screenshot -> scene point."""
    W, H = mob.px_size
    return mob.get_center() + np.array([(px / W - 0.5) * mob.width, (0.5 - py / H) * mob.height, 0])


# =====================================================================
# Synthetic batch used for the pictures in part 2
# (shapes are real, values are illustrative)
# =====================================================================
def make_fake_batch(seed=7):
    rng = np.random.default_rng(seed)
    x  = rng.uniform(-0.8, 0.8, N_ENVS)
    y  = rng.uniform(0.1, 1.4, N_ENVS)
    vx = rng.normal(0, 0.4, N_ENVS)
    vy = -np.abs(rng.normal(0.4, 0.3, N_ENVS))
    ang = rng.normal(0, 0.3, N_ENVS)
    angv = rng.normal(0, 0.2, N_ENVS)
    legs = (y < 0.25)[:, None] * (rng.uniform(size=(N_ENVS, 2)) < 0.7)
    states = np.column_stack([x, y, vx, vy, ang, angv, legs[:, 0], legs[:, 1]])

    W = rng.normal(0, 0.8, (N_STATE, N_ACTIONS))
    logits = states @ W
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    actions = np.array([rng.choice(N_ACTIONS, p=p) for p in probs])
    values = states @ rng.normal(0, 0.6, N_STATE)
    rewards = rng.normal(0, 1.0, N_ENVS)
    return dict(states=states, logits=logits, actions=actions, values=values, rewards=rewards)


BATCH = make_fake_batch()


class LessonPart:
    """One part of the video. Not a Manim Scene itself -> only one scene in the file."""
    pass


# =====================================================================
# Part 1: the task (state, actions, reward)
# =====================================================================
class Part01_Task(LessonPart):
    def construct(self):
        title = Text("Training A2C on Lunar Lander", font_size=50)
        sub = Text("128 parallel environments  ·  rollout buffer  ·  one update per buffer",
                   font_size=26, color=TEXT_DIM)
        VGroup(title, sub).arrange(DOWN, buff=0.3)
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(sub, shift=UP * 0.3))
        self.wait(1.5)
        self.play(FadeOut(sub), title.animate.scale(0.6).to_edge(UP, buff=0.35))

        # ---- screenshot ---------------------------------------------------
        shot = load_screenshot(height=4.6).move_to(DOWN * 0.3)
        self.play(FadeIn(shot, scale=0.95), run_time=1.2)
        self.wait(1)

        goal = Text("goal: land softly between the flags, using as little fuel as possible",
                    font_size=24).next_to(shot, DOWN, buff=0.25)
        notes = Group()
        if shot.px_size is not None:
            lander_pt = px_to_point(shot, 268, 62)
            ring = Circle(radius=0.42, color=YELLOW, stroke_width=3).move_to(lander_pt)
            l_lab = Text("lander", font_size=22, color=YELLOW).next_to(ring, RIGHT, buff=0.15)
            f_l, f_r = px_to_point(shot, 240, 300), px_to_point(shot, 360, 300)
            pad = Line(f_l, f_r, color=YELLOW, stroke_width=5)
            p_lab = Text("landing pad, at (0, 0)", font_size=22, color=YELLOW)\
                .next_to(pad, UP, buff=0.1).shift(UP * 0.75)
            self.play(Create(ring), FadeIn(l_lab))
            self.play(Create(pad), FadeIn(p_lab))
            notes = Group(ring, l_lab, pad, p_lab)
        self.play(FadeIn(goal))
        self.wait(2)

        # screenshot moves left, the right column explains the interfaces
        self.play(FadeOut(goal), FadeOut(notes),
                  shot.animate.set_height(3.4).move_to(LEFT * 4.1 + DOWN * 0.2), run_time=1.2)

        # ---- state: 8 numbers --------------------------------------------
        s_title = MathTex(r"\text{state } s \in \mathbb{R}^8", font_size=40, color=STATE_COLOR)
        rows = [
            ("s_0", "x", "horizontal position"),
            ("s_1", "y", "vertical position"),
            ("s_2", "v_x", "horizontal velocity"),
            ("s_3", "v_y", "vertical velocity"),
            ("s_4", r"\varphi", "angle"),
            ("s_5", r"\omega", "angular velocity"),
            ("s_6", "c_L", "left leg touches ground (0/1)"),
            ("s_7", "c_R", "right leg touches ground (0/1)"),
        ]
        table = VGroup()
        for idx, sym, desc in rows:
            table.add(VGroup(MathTex(idx, font_size=30, color=STATE_COLOR),
                             MathTex(sym, font_size=30),
                             Text(desc, font_size=20, color=TEXT_DIM)))
        cells = [m for r in table for m in r]
        VGroup(*cells).arrange_in_grid(rows=8, cols=3, buff=(0.4, 0.16), col_alignments="lll")
        s_block = fit_right_column(VGroup(s_title, table).arrange(DOWN, buff=0.3))

        self.play(Write(s_title))
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in table], lag_ratio=0.15),
                  run_time=2.5)
        shape_s = MathTex(r"\text{batched over 128 envs: } (128,\, 8)", font_size=28, color=STATE_COLOR)\
            .next_to(s_block, DOWN, buff=0.3)
        self.play(FadeIn(shape_s))
        self.wait(2.5)
        self.play(FadeOut(s_block), FadeOut(shape_s))

        # ---- actions: 4 discrete ---------------------------------------------
        a_title = Text("4 discrete actions", font_size=32)
        acts = VGroup()
        for k, (flame, name) in enumerate([(None, "do nothing"), ("left", "fire left engine"),
                                           ("main", "fire main engine"), ("right", "fire right engine")]):
            box = RoundedRectangle(corner_radius=0.12, width=1.6, height=1.25,
                                   stroke_color=ACTION_COLORS[k], stroke_width=3)
            icon = make_lander(0.55, flame).move_to(box)
            num = MathTex(f"a = {k}", font_size=30, color=ACTION_COLORS[k]).next_to(box, UP, buff=0.12)
            lab = Text(name, font_size=18, color=TEXT_DIM).next_to(box, DOWN, buff=0.12)
            acts.add(VGroup(num, box, icon, lab))
        acts.arrange_in_grid(rows=2, cols=2, buff=(0.7, 0.35))
        a_block = fit_right_column(VGroup(a_title, acts).arrange(DOWN, buff=0.35))

        self.play(Write(a_title))
        self.play(LaggedStart(*[FadeIn(a, scale=0.9) for a in acts], lag_ratio=0.25), run_time=2)
        shape_a = MathTex(r"\text{actor output: 4 logits} \;\rightarrow\; \text{batched } (128,\, 4)",
                          font_size=28, color=LOGIT_COLOR).next_to(a_block, DOWN, buff=0.3)
        self.play(FadeIn(shape_a))
        self.wait(2.5)
        self.play(FadeOut(a_block), FadeOut(shape_a))

        # ---- reward -------------------------------------------------------
        r_title = Text("reward r (per step)", font_size=32, color=REWARD_COLOR)
        r_lines = VGroup(
            Text("+  moving closer to the pad, slowing down, staying level", font_size=21),
            Text("+10  for each leg touching the ground", font_size=21),
            Text("−  firing engines costs fuel (main engine more)", font_size=21),
            Text("+100  landed and came to rest", font_size=21, color=GREEN_C),
            Text("−100  crashed or flew away", font_size=21, color=RED_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        solved = Text("episode return ≥ 200  →  considered solved", font_size=22, color=YELLOW)
        r_block = fit_right_column(VGroup(r_title, r_lines, solved).arrange(DOWN, buff=0.4))
        self.play(Write(r_title))
        self.play(LaggedStart(*[FadeIn(l, shift=LEFT * 0.2) for l in r_lines], lag_ratio=0.3), run_time=2.5)
        self.play(FadeIn(solved))
        self.wait(3)

        nxt = Text("Next: run 128 of these at once", font_size=24, color=STATE_COLOR)\
            .to_edge(DOWN, buff=0.35)
        self.play(FadeIn(nxt, shift=UP * 0.2))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not shot], run_time=0.8)
        self.lunar_shot = shot          # handed over to part 2 (if rendered together)


# =====================================================================
# Part 2: 128 parallel environments, one batched forward pass
# =====================================================================
class Part02_ParallelEnvs(LessonPart):
    def construct(self):
        B = BATCH
        header = section_header("128 parallel environments", STATE_COLOR)
        code = Text("envs = VecEnv(n_envs=128)", font="Monospace", font_size=24, color=TEXT_DIM)\
            .next_to(header, DOWN, buff=0.2)

        # ---- grid of tiles --------------------------------------------------
        tw, th, gap = 0.78, 0.54, 0.05
        tiles = VGroup(*[make_tile(tw, th) for _ in range(N_ENVS)])
        tiles.arrange_in_grid(rows=GRID_ROWS, cols=GRID_COLS, buff=gap)
        tiles.next_to(code, DOWN, buff=0.35)

        # per-env motion: episodes of different length, resets at different times
        rng = np.random.default_rng(3)
        ep_len = rng.uniform(4.0, 9.0, N_ENVS)
        offset = rng.uniform(0, 9.0, N_ENVS)
        x0 = rng.uniform(-0.25, 0.25, N_ENVS)
        drift = rng.normal(0, 0.12, N_ENVS)
        wob_f = rng.uniform(1.0, 2.5, N_ENVS)
        wob_a = rng.uniform(0.05, 0.35, N_ENVS)

        clock = ValueTracker(0.0)
        base_lander = make_lander(1.0)

        def phase(i):
            return ((clock.get_value() + offset[i]) % ep_len[i]) / ep_len[i]   # 0..1

        def lander_updater(i):
            def upd(m):
                tile = tiles[i]
                sc = tile.width / tw
                p = phase(i)
                rel = np.array([(x0[i] + drift[i] * p) * tw, (0.33 - 0.55 * p) * th, 0])
                m.become(base_lander.copy().scale_to_fit_height(0.17 * sc)
                         .rotate(wob_a[i] * np.sin(wob_f[i] * 6.28 * p))
                         .move_to(tile.get_center() + rel * sc))
            return upd

        def tile_updater(i):
            def upd(t):
                fresh = phase(i) < 0.06         # episode just restarted
                t[0].set_stroke(DONE_COLOR if fresh else GREY_D, width=3 if fresh else 1.5)
            return upd

        landers = VGroup()
        for i in range(N_ENVS):
            m = base_lander.copy()
            lander_updater(i)(m)
            landers.add(m)

        # transition from part 1: the screenshot shrinks into tile 0
        shot = getattr(self, "lunar_shot", None)
        self.play(FadeIn(header, shift=DOWN * 0.2))
        if shot is not None:
            self.play(shot.animate.set_height(th).move_to(tiles[0]), run_time=1.3)
            self.play(FadeIn(tiles[0]), FadeIn(landers[0]), FadeOut(shot), run_time=0.5)
            self.lunar_shot = None
        else:
            self.play(FadeIn(tiles[0]), FadeIn(landers[0]))
        self.play(Write(code))
        self.play(LaggedStart(*[FadeIn(VGroup(tiles[i], landers[i]), scale=0.6)
                                for i in range(1, N_ENVS)], lag_ratio=0.02), run_time=2.5)

        for i in range(N_ENVS):
            landers[i].add_updater(lander_updater(i))
            tiles[i].add_updater(tile_updater(i))

        note = Text("independent episodes: each env restarts on its own (yellow frame = reset)",
                    font_size=22, color=TEXT_DIM).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(note), clock.animate.increment_value(1.5), run_time=1.5, rate_func=linear)
        self.play(clock.animate.increment_value(4.0), run_time=4.0, rate_func=linear)

        # ---- shrink grid to the left, build the batched pipeline ----------------
        self.play(FadeOut(note), FadeOut(code),
                  tiles.animate.scale(0.4).move_to(LEFT * 4.35 + UP * 1.1),
                  clock.animate.increment_value(1.2), run_time=1.2, rate_func=linear)
        envs_lab = Text("128 envs", font_size=22, color=STATE_COLOR).next_to(tiles, DOWN, buff=0.1)
        self.play(FadeIn(envs_lab), clock.animate.increment_value(0.6), run_time=0.6, rate_func=linear)

        H = 3.2                                        # height of all (128, ...) tensors
        y_t = UP * 0.55
        st_img = heatmap(B["states"], STATE_COLOR, H, 0.8).move_to(LEFT * 1.15 + y_t)
        st_lab = VGroup(Text("states", font_size=22, color=STATE_COLOR),
                        MathTex("(128,\\,8)", font_size=28)).arrange(DOWN, buff=0.08)\
            .next_to(st_img, DOWN, buff=0.15)

        mlp_box = RoundedRectangle(corner_radius=0.15, width=2.2, height=2.4,
                                   stroke_color=WHITE, stroke_width=2.5, fill_opacity=0.05)\
            .move_to(RIGHT * 1.15 + y_t)
        mlp_txt = VGroup(Text("MLPRLModel", font_size=22),
                         MathTex(r"8 \to 128 \to 128", font_size=28),
                         Text("SiLU", font_size=18, color=TEXT_DIM),
                         MathTex(r"\text{actor: } 4", font_size=26, color=LOGIT_COLOR),
                         MathTex(r"\text{critic: } 1", font_size=26, color=VALUE_COLOR))\
            .arrange(DOWN, buff=0.12).move_to(mlp_box)

        lg_img = heatmap(B["logits"], LOGIT_COLOR, H, 0.55, diverging=True).move_to(RIGHT * 3.2 + y_t)
        lg_lab = VGroup(Text("logits", font_size=22, color=LOGIT_COLOR),
                        MathTex("(128,\\,4)", font_size=28)).arrange(DOWN, buff=0.08)\
            .next_to(lg_img, DOWN, buff=0.15)
        v_img = heatmap(B["values"][:, None], VALUE_COLOR, H, 0.16).next_to(lg_img, RIGHT, buff=0.75)
        v_lab = VGroup(Text("V", font_size=22, color=VALUE_COLOR),
                       MathTex("(128,\\,1)", font_size=28)).arrange(DOWN, buff=0.08)\
            .next_to(v_img, DOWN, buff=0.15).align_to(lg_lab, UP)

        ac_img = categorical_strip(B["actions"], ACTION_COLORS, H, 0.2).move_to(RIGHT * 5.9 + y_t)
        ac_lab = VGroup(Text("actions", font_size=22),
                        MathTex("(128,)", font_size=28)).arrange(DOWN, buff=0.08)\
            .next_to(ac_img, DOWN, buff=0.15).align_to(lg_lab, UP)

        arr1 = Arrow(st_img.get_right(), mlp_box.get_left(), buff=0.1, color=STATE_COLOR)
        arr2 = Arrow(mlp_box.get_right(), lg_img.get_left(), buff=0.1, color=LOGIT_COLOR)
        arr3 = Arrow(v_img.get_right(), ac_img.get_left(), buff=0.12, color=PROB_COLOR)
        samp = Text("sample", font_size=18, color=PROB_COLOR).next_to(arr3, DOWN, buff=0.1)

        # each env sends its state row: 128 dots fly from the tiles into the states tensor
        def row_y(i):
            return st_img.get_top()[1] - (i + 0.5) * H / N_ENVS
        dots = VGroup(*[Square(0.05, fill_color=STATE_COLOR, fill_opacity=1, stroke_width=0)
                        .move_to(tiles[i].get_center()) for i in range(N_ENVS)])
        self.add(dots)
        self.play(LaggedStart(*[dots[i].animate.move_to([st_img.get_x(), row_y(i), 0])
                                for i in range(N_ENVS)], lag_ratio=0.01),
                  clock.animate.increment_value(1.6), run_time=1.6, rate_func=linear)
        self.play(FadeIn(st_img), FadeOut(dots), FadeIn(st_lab), run_time=0.8)
        stack_note = Text("stack: row i = state of env i", font_size=22, color=TEXT_DIM)\
            .to_edge(DOWN, buff=0.35)
        self.play(FadeIn(stack_note))
        self.wait(1)

        # forward pass
        self.play(GrowArrow(arr1), FadeIn(mlp_box), FadeIn(mlp_txt), run_time=1.2)
        self.play(GrowArrow(arr2), FadeIn(lg_img), FadeIn(lg_lab), FadeIn(v_img), FadeIn(v_lab), run_time=1.2)
        fwd_note = Text("ONE forward pass for all 128 envs:  logits, V = model(states)",
                        font_size=22).to_edge(DOWN, buff=0.35)
        self.play(ReplacementTransform(stack_note, fwd_note))
        self.wait(1.5)
        self.play(GrowArrow(arr3), Write(samp), FadeIn(ac_img), FadeIn(ac_lab), run_time=1.2)

        legend = VGroup(*[VGroup(Square(0.18, fill_color=ACTION_COLORS[k], fill_opacity=1, stroke_width=0),
                                 MathTex(str(k), font_size=24)).arrange(RIGHT, buff=0.08)
                          for k in range(N_ACTIONS)]).arrange(DOWN, buff=0.2, aligned_edge=LEFT)\
            .next_to(ac_img, RIGHT, buff=0.25)
        self.play(FadeIn(legend))
        self.wait(1)

        # ---- follow env 0 through the pipeline -------------------------------
        follow = Text("follow env 0", font_size=22, color=YELLOW).to_edge(DOWN, buff=0.35)
        self.play(ReplacementTransform(fwd_note, follow))

        def row_box(img):
            return Rectangle(width=img.width + 0.1, height=0.09, color=YELLOW, stroke_width=3)\
                .move_to([img.get_x(), img.get_top()[1] - H / N_ENVS / 2, 0])
        t0_box = SurroundingRectangle(tiles[0], color=YELLOW, buff=0.03, stroke_width=3)
        boxes = [row_box(st_img), row_box(lg_img), row_box(v_img), row_box(ac_img)]
        self.play(Create(t0_box))
        for b in boxes:
            self.play(Create(b), run_time=0.5)
        a0 = int(B["actions"][0])
        a0_txt = MathTex(f"a = {a0}", font_size=30, color=ACTION_COLORS[a0])\
            .next_to(boxes[3], RIGHT, buff=0.1)
        self.play(FadeIn(a0_txt))
        self.wait(1.5)
        self.play(FadeOut(VGroup(t0_box, *boxes, a0_txt)))

        # ---- envs.step(actions) ---------------------------------------------
        # return path over the top: actions -> up -> left -> down into the env grid
        y_top = 2.72
        p0 = ac_img.get_top() + UP * 0.08
        p1 = [p0[0], y_top, 0]
        p2 = [tiles.get_x(), y_top, 0]
        p3 = tiles.get_top() + UP * 0.06
        back_path = VMobject(color=WHITE, stroke_width=3).set_points_as_corners([p0, p1, p2])
        back_tip = Arrow(p2, p3, buff=0, color=WHITE, stroke_width=3, max_tip_length_to_length_ratio=0.5)
        back = VGroup(back_path, back_tip)
        back_lab = Text("envs.step(actions)", font="Monospace", font_size=20)\
            .next_to(Line(p1, p2), DOWN, buff=0.1).shift(RIGHT * 1.2)

        self.play(FadeOut(follow), Create(back_path), run_time=0.9)
        self.play(GrowArrow(back_tip), FadeIn(back_lab), run_time=0.5)
        # actions travel back to the envs
        adots = VGroup(*[Square(0.05, fill_color=ACTION_COLORS[int(B["actions"][i])], fill_opacity=1,
                                stroke_width=0).move_to([ac_img.get_x(), row_y(i), 0])
                         for i in range(N_ENVS)])
        self.add(adots)
        self.play(LaggedStart(*[adots[i].animate.move_to(tiles[i].get_center()) for i in range(N_ENVS)],
                              lag_ratio=0.008),
                  clock.animate.increment_value(1.6), run_time=1.6, rate_func=linear)
        self.play(FadeOut(adots), run_time=0.3)

        # rewards and dones come back, one entry per env
        rw_img = heatmap(B["rewards"][None, :], REWARD_COLOR, 0.16, tiles.width, diverging=True)
        rw_img.next_to(envs_lab, DOWN, buff=0.5)
        rw_lab = VGroup(Text("rewards", font_size=20, color=REWARD_COLOR),
                        MathTex("(128,)", font_size=24)).arrange(RIGHT, buff=0.1)\
            .next_to(rw_img, UP, buff=0.08).align_to(rw_img, LEFT)
        mask = np.array([phase(i) < 0.06 for i in range(N_ENVS)])   # envs that just reset
        dn_img = heatmap(mask[None, :].astype(float), DONE_COLOR, 0.16, tiles.width, lo=0, hi=1)
        dn_img.next_to(rw_img, DOWN, buff=0.42)
        dn_lab = VGroup(Text("dones", font_size=20, color=DONE_COLOR),
                        MathTex("(128,)", font_size=24)).arrange(RIGHT, buff=0.1)\
            .next_to(dn_img, UP, buff=0.08).align_to(dn_img, LEFT)
        ret_txt = Text("returns new states, rewards, dones", font_size=22, color=TEXT_DIM)\
            .to_edge(DOWN, buff=0.35)
        self.play(FadeIn(rw_img), FadeIn(rw_lab), FadeIn(dn_img), FadeIn(dn_lab), FadeIn(ret_txt), run_time=1.0)
        done_note = Text("done = 1 where an episode just ended", font_size=18, color=DONE_COLOR)\
            .next_to(dn_img, DOWN, buff=0.12).align_to(dn_img, LEFT)
        self.play(FadeIn(done_note))
        self.wait(1.5)

        # ---- the loop runs: a few fast iterations ------------------------------
        counter_val = ValueTracker(1)
        counter = always_redraw(lambda: MathTex(rf"\text{{env step }} {int(counter_val.get_value())}",
                                                font_size=30, color=YELLOW)
                                .next_to(tiles, UP, buff=0.12).align_to(tiles, LEFT))
        self.play(FadeOut(ret_txt), FadeIn(counter))
        loop_arrows = [arr1, arr2, arr3, back_path]
        for k in range(4):
            self.play(*[ShowPassingFlash(a.copy().set_color(YELLOW).set_stroke(width=6), time_width=0.6)
                        for a in loop_arrows],
                      counter_val.animate.set_value(k + 2),
                      clock.animate.increment_value(0.9), run_time=0.9, rate_func=linear)

        take = VGroup(
            Text("all 128 envs advance together, one network call per step", font_size=24),
            Text("next: store every step in the rollout buffer, 128 steps × 128 envs",
                 font_size=22, color=TEXT_DIM),
        ).arrange(DOWN, buff=0.12).to_edge(DOWN, buff=0.25)
        self.play(FadeOut(done_note), FadeOut(back_lab), FadeIn(take, shift=UP * 0.2),
                  clock.animate.increment_value(1.0), run_time=1.0, rate_func=linear)
        self.play(clock.animate.increment_value(3.0), run_time=3.0, rate_func=linear)

        for i in range(N_ENVS):
            landers[i].clear_updaters()
            tiles[i].clear_updaters()
        counter.clear_updaters()
        self.play(*[FadeOut(m) for m in self.mobjects])


# =====================================================================
# Synthetic rollout (128 steps x 128 envs) for parts 3 - 5
# Shapes and the return recursion are exactly as in a2c_agent.py,
# the values are illustrative.
# =====================================================================
RETURN_COLOR = "#B983FF"     # violet -> returns R
ADV_COLOR    = YELLOW        # advantages
POS_COLOR    = GREEN_C       # positive reward / return
GAMMA        = 0.99
T_STEPS      = 128


def make_fake_rollout(seed=11, T=T_STEPS, N=N_ENVS, gamma=GAMMA):
    rng = np.random.default_rng(seed)
    states  = np.zeros((T, N, N_STATE))
    actions = np.zeros((T, N), dtype=int)
    rewards = np.zeros((T, N))
    dones   = np.zeros((T, N), dtype=bool)

    age    = rng.integers(0, 160, N)            # steps since episode start
    length = rng.integers(70, 230, N)           # episode length
    x_dir  = rng.uniform(-1, 1, N)
    pref   = rng.dirichlet(np.ones(N_ACTIONS) * 2, N)

    for t in range(T):
        p = np.clip(age / length, 0, 1)
        states[t, :, 0] = x_dir * (1 - p) * 0.8
        states[t, :, 1] = 1.4 * (1 - p)
        states[t, :, 2] = -x_dir * 0.3
        states[t, :, 3] = -0.6 * (1 - p)
        states[t, :, 4] = 0.2 * np.sin(6 * p + x_dir)
        states[t, :, 5] = 0.1 * np.cos(6 * p)
        states[t, :, 6] = p > 0.95
        states[t, :, 7] = p > 0.97
        actions[t] = [rng.choice(N_ACTIONS, p=pr) for pr in pref]
        r = rng.normal(0, 0.8, N) - 0.3 * (actions[t] == 2)
        done = age + 1 >= length
        landed = rng.uniform(size=N) < 0.5
        r[done] = np.where(landed[done], 100.0, -100.0)
        rewards[t], dones[t] = r, done
        age = np.where(done, 0, age + 1)
        length = np.where(done, rng.integers(70, 230, N), length)
        pref = np.where(done[:, None], rng.dirichlet(np.ones(N_ACTIONS) * 2, N), pref)

    # n-step returns, the same loop as in a2c_agent.py
    next_value = rng.normal(10, 8, N)
    returns = np.zeros((T, N))
    R = next_value.copy()
    for n in reversed(range(T)):
        R = rewards[n] + gamma * R * (1.0 - dones[n])
        returns[n] = R

    values = 0.8 * returns + rng.normal(0, 8, (T, N))     # an imperfect critic
    adv = returns - values
    return dict(states=states, actions=actions, rewards=rewards, dones=dones,
                next_value=next_value, returns=returns, values=values, advantages=adv)


ROLLOUT = make_fake_rollout()


def categorical_image(cats, colors, height, width):
    """2D array of discrete values -> image, one colored pixel per entry."""
    pal = np.array([color_to_rgb(c) for c in colors])
    img = ImageMobject((pal[np.asarray(cats, dtype=int)] * 255).astype(np.uint8))
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    img.stretch_to_fit_height(height)
    img.stretch_to_fit_width(width)
    return img


def stacked_layers(img, depth=7, step=0.05, color=STATE_COLOR):
    """Outline layers behind an image, to suggest a third tensor dimension."""
    return VGroup(*[Rectangle(width=img.width, height=img.height, stroke_color=color, stroke_width=1,
                              stroke_opacity=0.7, fill_color="#0b2a28", fill_opacity=1)
                    .move_to(img.get_center() + k * np.array([step, step, 0]))
                    for k in range(depth, 0, -1)])


def fmt(x, nd=2):
    """Number for LaTeX, negatives wrapped in parentheses."""
    s = f"{x:.{nd}f}"
    return f"({s})" if x < 0 else s


def tensor_label(name, shape, color, font=22):
    return VGroup(Text(name, font_size=font, color=color),
                  MathTex(shape, font_size=font + 4)).arrange(RIGHT, buff=0.12)


def cell_point(img, t, n, T=T_STEPS, N=N_ENVS):
    """Scene point of entry (t, n) inside a (T, N) image."""
    return np.array([img.get_left()[0] + (n + 0.5) / N * img.width,
                     img.get_top()[1] - (t + 0.5) / T * img.height, 0])


def code_block(lines, font_size=18):
    return VGroup(*[Text(l, font="Monospace", font_size=font_size) for l in lines])\
        .arrange(DOWN, aligned_edge=LEFT, buff=0.1)


# =====================================================================
# Part 3: filling the rollout buffer
# =====================================================================
class Part03_RolloutBuffer(LessonPart):
    def construct(self):
        R = ROLLOUT
        T, N = T_STEPS, N_ENVS
        header = section_header("Rollout buffer: 128 steps × 128 envs", STATE_COLOR)
        sub = Text("rows = time steps t (0 … 127),   columns = environments (0 … 127)",
                   font_size=22, color=TEXT_DIM).next_to(header, DOWN, buff=0.2)
        self.play(FadeIn(header, shift=DOWN * 0.2))
        self.play(FadeIn(sub))

        P, yc = 2.6, -0.25
        xs = [-4.85, -1.6, 1.65, 4.9]
        st = heatmap(R["states"][:, :, 1], STATE_COLOR, P, P).move_to([xs[0], yc, 0])
        layers = stacked_layers(st)
        ac = categorical_image(R["actions"], ACTION_COLORS, P, P).move_to([xs[1], yc, 0])
        rw = heatmap(R["rewards"], POS_COLOR, P, P, lo=-10, hi=10, diverging=True,
                     neg_color=REWARD_COLOR).move_to([xs[2], yc, 0])
        dn = heatmap(R["dones"].astype(float), DONE_COLOR, P, P, lo=0, hi=1).move_to([xs[3], yc, 0])
        imgs = [st, ac, rw, dn]

        labels = VGroup(
            tensor_label("states", "(128,128,8)", STATE_COLOR, 20).next_to(layers, UP, buff=0.12),
            tensor_label("actions", "(128,128)", WHITE, 20).next_to(ac, UP, buff=0.12),
            tensor_label("rewards", "(128,128)", POS_COLOR, 20).next_to(rw, UP, buff=0.12),
            tensor_label("dones", "(128,128)", DONE_COLOR, 20).next_to(dn, UP, buff=0.12),
        )
        labels[1:].align_to(labels[0], DOWN)
        frames = VGroup(*[Rectangle(width=P, height=P, stroke_color=GREY_C, stroke_width=1.5)
                          .move_to(im) for im in imgs])
        st_note = Text("shown: channel y of the 8", font_size=16, color=TEXT_DIM)\
            .next_to(st, DOWN, buff=0.1)

        k = ValueTracker(0.0)

        def make_cover(img):
            c = Rectangle(width=img.width + 0.02, height=P, fill_color=BLACK, fill_opacity=1, stroke_width=0)

            def upd(m):
                frac = 1 - k.get_value() / T
                h = max(frac * P, 1e-3)
                m.stretch_to_fit_height(h)
                m.move_to([img.get_x(), img.get_bottom()[1] + h / 2, 0])
                m.set_fill(opacity=1 if frac > 1e-3 else 0)
            upd(c)
            c.add_updater(upd)
            return c

        def make_cursor(img):
            line = Line(LEFT, RIGHT, color=YELLOW, stroke_width=3)

            def upd(m):
                row = min(int(k.get_value()), T - 1)
                y = img.get_top()[1] - (row + 0.5) / T * P
                m.put_start_and_end_on([img.get_left()[0] - 0.08, y, 0], [img.get_right()[0] + 0.08, y, 0])
                m.set_stroke(opacity=0 if k.get_value() >= T else 1)
            upd(line)
            line.add_updater(upd)
            return line

        covers = [make_cover(im) for im in imgs]
        self.add(layers, *imgs, *covers)
        self.play(FadeIn(frames), FadeIn(layers), FadeIn(labels), FadeIn(st_note))

        code = code_block([
            "# a2c_agent.step(), every env step:",
            "self.states [step_idx] = state     # (128, 8)",
            "self.actions[step_idx] = actions   # (128,)",
            "self.rewards[step_idx] = reward    # (128,)",
            "self.dones  [step_idx] = done      # (128,)",
            "step_idx += 1",
        ], font_size=17).to_edge(DOWN, buff=0.3).to_edge(RIGHT, buff=0.5)
        code[0].set_color(TEXT_DIM)
        counter = always_redraw(lambda: MathTex(rf"\texttt{{step\_idx}} = {min(int(k.get_value()), T)}",
                                                font_size=32, color=YELLOW)
                                .move_to(LEFT * 4.6 + DOWN * 2.4))
        self.play(FadeIn(code), FadeIn(counter))
        cursors = [make_cursor(im) for im in imgs]
        self.add(*cursors)
        self.wait(1)

        row_note = Text("one env step = one new row in all four tensors", font_size=19)\
            .move_to(LEFT * 3.85 + DOWN * 3.0)
        self.play(FadeIn(row_note))
        for i in range(4):
            self.play(k.animate.set_value(i + 1), run_time=0.7, rate_func=smooth)
            self.wait(0.3)
        self.play(FadeOut(row_note))
        self.play(k.animate.set_value(T), run_time=5, rate_func=linear)
        self.wait(0.5)

        # ---- buffer full ----------------------------------------------------
        full = VGroup(
            Text("buffer full:  128 × 128 = 16,384 transitions", font_size=26, color=YELLOW),
            Text("→ train once, then clear it (_reset_buffer) and collect again", font_size=22),
        ).arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.85)
        self.play(FadeOut(code), FadeOut(counter), FadeIn(full, shift=UP * 0.2))
        self.wait(2)

        # ---- point out a few episode ends -------------------------------------
        ends = np.argwhere(R["dones"])
        inner = [e for e in ends if 15 < e[0] < 112 and 12 < e[1] < 116]
        picks = [inner[i] for i in np.linspace(0, len(inner) - 1, 3).astype(int)]
        rings = VGroup()
        for (tt, nn) in picks:
            for im in (rw, dn):
                rings.add(Circle(radius=0.11, color=WHITE, stroke_width=3).move_to(cell_point(im, tt, nn)))
        end_note = Text("episode ended (reward ±100): the env resets, and its column continues "
                        "with a new episode", font_size=20, color=DONE_COLOR).to_edge(DOWN, buff=0.3)
        self.play(Create(rings), FadeIn(end_note))
        self.wait(3)

        for m in covers + cursors:
            m.clear_updaters()
        self.play(*[FadeOut(m) for m in self.mobjects])


# =====================================================================
# Part 4: n-step returns, computed backwards
# =====================================================================
class Part04_Returns(LessonPart):
    def construct(self):
        RO = ROLLOUT
        header = section_header("n-step returns, computed backwards", RETURN_COLOR)
        self.play(FadeIn(header, shift=DOWN * 0.2))

        eq = MathTex("R_t", "=", "r_t", "+", r"\gamma\,", "R_{t+1}", r"\,(1 - d_t)", font_size=48)
        eq[0].set_color(RETURN_COLOR)
        eq[2].set_color(POS_COLOR)
        eq[5].set_color(RETURN_COLOR)
        eq[6].set_color(DONE_COLOR)
        eq.move_to(UP * 1.2)
        expl = VGroup(
            Text("return = reward now + discounted future", font_size=24),
            Text("the future is cut where the episode ended (d = 1)", font_size=24, color=DONE_COLOR),
            MathTex(r"\gamma = 0.99, \qquad R_T = V(s_{last}) \;\; \text{(bootstrap from the critic)}",
                    font_size=32, color=TEXT_DIM),
        ).arrange(DOWN, buff=0.25).next_to(eq, DOWN, buff=0.5)
        self.play(Write(eq))
        for e in expl:
            self.play(FadeIn(e, shift=UP * 0.15))
            self.wait(0.8)
        self.wait(1)

        eq_small = VGroup(eq).copy().scale(0.75).move_to(RIGHT * 3.4 + UP * 2.2)
        self.play(FadeOut(expl), Transform(eq, eq_small[0]))

        # ---- one env column, last 8 steps -------------------------------
        ts = list(range(120, 128))
        rs = [1.3, -0.4, 2.1, 100.0, -0.6, 0.8, 1.5, -0.3]
        ds = [0, 0, 0, 1, 0, 0, 0, 0]
        v_last = 12.40
        Rs = [0.0] * 8
        Rn = v_last
        for i in reversed(range(8)):
            Rn = rs[i] + GAMMA * Rn * (1 - ds[i])
            Rs[i] = Rn

        col_x = [-5.7, -4.4, -3.2, -1.7]
        y0, dy = 1.75, 0.45
        head = VGroup(MathTex("t", font_size=32), MathTex("r_t", font_size=32, color=POS_COLOR),
                      MathTex("d_t", font_size=32, color=DONE_COLOR),
                      MathTex("R_t", font_size=32, color=RETURN_COLOR))
        for j, h in enumerate(head):
            h.move_to([col_x[j], y0 + 0.5, 0])
        rows = VGroup()
        for i in range(8):
            y = y0 - i * dy
            rows.add(VGroup(MathTex(str(ts[i]), font_size=30).move_to([col_x[0], y, 0]),
                            MathTex(f"{rs[i]:.1f}", font_size=30).move_to([col_x[1], y, 0]),
                            MathTex(str(ds[i]), font_size=30,
                                    color=DONE_COLOR if ds[i] else WHITE).move_to([col_x[2], y, 0])))
        y_boot = y0 - 8 * dy - 0.15
        sep = Line([col_x[0] - 0.5, y_boot + 0.27, 0], [col_x[3] + 0.6, y_boot + 0.27, 0], color=GREY_C)
        boot_lab = MathTex(r"V(s_{last})", font_size=30, color=VALUE_COLOR).move_to([col_x[1] + 0.3, y_boot, 0])
        boot_val = MathTex(f"{v_last:.2f}", font_size=30, color=VALUE_COLOR).move_to([col_x[3], y_boot, 0])
        done_bg = Rectangle(width=col_x[3] - col_x[0] + 1.2, height=dy * 0.95, stroke_width=0,
                            fill_color=DONE_COLOR, fill_opacity=0.15)\
            .move_to([(col_x[0] + col_x[3]) / 2, y0 - 3 * dy, 0])
        table = VGroup(head, rows, sep, boot_lab, boot_val)

        tag = Text("one env, last 8 steps of the buffer", font_size=20, color=TEXT_DIM)\
            .next_to(head, UP, buff=0.25).align_to(head, LEFT)
        self.play(FadeIn(tag), FadeIn(head), LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.1),
                  run_time=1.5)
        self.play(FadeIn(done_bg))
        self.play(Create(sep), FadeIn(boot_lab), FadeIn(boot_val))
        boot_note = Text("start: the critic's guess for what comes after the buffer",
                         font_size=20, color=VALUE_COLOR).move_to(RIGHT * 3.0 + DOWN * 2.3)
        self.play(FadeIn(boot_note))
        self.wait(1.5)

        # ---- the backward sweep -----------------------------------------------
        cursor = SurroundingRectangle(boot_val, color=YELLOW, buff=0.08)
        self.play(Create(cursor), FadeOut(boot_note))
        prev_cell, comp, cut_note = boot_val, None, None
        R_cells = VGroup()
        for step, i in enumerate(reversed(range(8))):
            y = y0 - i * dy
            Rnext = Rs[i + 1] if i < 7 else v_last
            new_comp = MathTex(f"R_{{{ts[i]}}}", "=", fmt(rs[i], 1), "+", r"0.99 \cdot",
                               f"{fmt(Rnext, 2)}", rf"\cdot (1 - {ds[i]})", "=", f"{Rs[i]:.2f}",
                               font_size=32).move_to(RIGHT * 3.4 + UP * 0.6)
            new_comp[0].set_color(RETURN_COLOR)
            new_comp[2].set_color(POS_COLOR)
            new_comp[5].set_color(RETURN_COLOR)
            new_comp[6].set_color(DONE_COLOR)
            new_comp[8].set_color(RETURN_COLOR)
            if new_comp.width > 6.6:
                new_comp.scale_to_fit_width(6.6)
            cell = MathTex(f"{Rs[i]:.2f}", font_size=30, color=RETURN_COLOR).move_to([col_x[3], y, 0])
            arc = CurvedArrow(prev_cell.get_right() + RIGHT * 0.1, cell.get_right() + RIGHT * 0.1,
                              angle=PI / 2, color=GREY_B if ds[i] == 0 else RED_C, stroke_width=2.5,
                              tip_length=0.15)
            slow = step < 3 or ds[i] == 1
            rt = 1.0 if slow else 0.45
            anims = [cursor.animate.move_to(VGroup(rows[i], cell)).stretch_to_fit_width(col_x[3] - col_x[0] + 1.1)]
            if comp is None:
                anims.append(Write(new_comp))
            else:
                anims.append(ReplacementTransform(comp, new_comp))
            self.play(*anims, run_time=rt)
            comp = new_comp
            self.play(Create(arc), TransformFromCopy(comp[8], cell), run_time=rt)
            if ds[i] == 1:
                cross = MathTex(r"\times", font_size=48, color=RED_C).move_to(arc.point_from_proportion(0.5))
                cut_note = VGroup(Text("done: the future is cut,  R = r", font_size=24, color=DONE_COLOR),
                                  Text("the next episode's rewards do not leak into this one",
                                       font_size=20, color=TEXT_DIM)).arrange(DOWN, buff=0.12)\
                    .next_to(comp, DOWN, buff=0.5)
                self.play(FadeIn(cross), FadeIn(cut_note))
                self.wait(2.5)
                self.play(FadeOut(cut_note))
                R_cells.add(cross)
            elif slow:
                self.wait(0.8)
            R_cells.add(cell, arc)
            prev_cell = cell

        prop = VGroup(Text("the +100 landing reward flows back", font_size=22, color=POS_COLOR),
                      Text("to the earlier steps 120 … 122", font_size=22, color=POS_COLOR))\
            .arrange(DOWN, buff=0.1).move_to(RIGHT * 3.4 + DOWN * 0.5)
        self.play(FadeIn(prop))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in (table, done_bg, R_cells, cursor, comp, prop, tag)])

        # ---- all 128 columns at once --------------------------------------------
        P, yc = 3.2, -0.15
        rw = heatmap(RO["rewards"], POS_COLOR, P, P, lo=-10, hi=10, diverging=True,
                     neg_color=REWARD_COLOR).move_to([-4.6, yc, 0])
        rt_img = heatmap(RO["returns"], POS_COLOR, P, P, lo=-60, hi=60, diverging=True,
                         neg_color=REWARD_COLOR).move_to([-0.9, yc, 0])
        frames = VGroup(*[Rectangle(width=P, height=P, stroke_color=GREY_C, stroke_width=1.5).move_to(m)
                          for m in (rw, rt_img)])
        lab_rw = tensor_label("rewards", "(128,128)", POS_COLOR, 20).next_to(rw, UP, buff=0.15)
        lab_rt = tensor_label("returns", "(128,128)", RETURN_COLOR, 20).next_to(rt_img, UP, buff=0.15)
        arrow = Arrow(rw.get_right(), rt_img.get_left(), buff=0.1, color=RETURN_COLOR)

        sweep = ValueTracker(0.0)
        cover = Rectangle(width=P + 0.02, height=P, fill_color=BLACK, fill_opacity=1, stroke_width=0)

        def cov_upd(m):
            frac = 1 - sweep.get_value()
            h = max(frac * P, 1e-3)
            m.stretch_to_fit_height(h)
            m.move_to([rt_img.get_x(), rt_img.get_top()[1] - h / 2, 0])      # shrinks toward the top
            m.set_fill(opacity=1 if frac > 1e-3 else 0)
        cov_upd(cover)
        cover.add_updater(cov_upd)

        code = code_block([
            "R = next_value                 # (128,)",
            "for n in reversed(range(T)):",
            "    R = rewards[n] + gamma * R * (1 - dones[n])",
            "    returns[n] = R",
        ], font_size=17)
        if code.width > 5.0:
            code.scale_to_fit_width(5.0)
        code.move_to(RIGHT * 4.15 + UP * 0.6)
        par = VGroup(Text("one loop over time,", font_size=22),
                     Text("all 128 envs in parallel:", font_size=22),
                     Text("R is a vector of 128", font_size=22, color=RETURN_COLOR))\
            .arrange(DOWN, buff=0.1).next_to(code, DOWN, buff=0.5)

        self.add(rt_img, cover)
        self.play(FadeIn(rw), FadeIn(frames), FadeIn(lab_rw), FadeIn(lab_rt), GrowArrow(arrow), FadeIn(code))
        self.play(FadeIn(par))
        self.play(sweep.animate.set_value(1.0), run_time=4, rate_func=linear)
        streak = Text("big rewards become streaks reaching back in time",
                      font_size=22, color=TEXT_DIM).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(streak))
        self.wait(3)
        cover.clear_updaters()
        self.play(*[FadeOut(m) for m in self.mobjects])


# =====================================================================
# Part 5: one big batch, advantages and their normalization
# =====================================================================
def histogram_bars(values, axis, edges, color, max_h):
    v = np.clip(values, edges[0] + 1e-9, edges[-1] - 1e-9)
    counts, _ = np.histogram(v, bins=edges)
    scale = max_h / counts.max()
    bars = VGroup()
    for c, a, b in zip(counts, edges[:-1], edges[1:]):
        x0, x1 = axis.n2p(a)[0], axis.n2p(b)[0]
        h = max(c * scale, 1e-3)
        bars.add(Rectangle(width=x1 - x0, height=h, fill_color=color, fill_opacity=0.85,
                           stroke_color=BLACK, stroke_width=1)
                 .move_to([(x0 + x1) / 2, axis.get_y() + h / 2, 0]))
    return bars


class Part05_BigBatch(LessonPart):
    def construct(self):
        RO = ROLLOUT
        T, N = T_STEPS, N_ENVS
        header = section_header("One big batch: 16,384 transitions", LOGIT_COLOR)
        self.play(FadeIn(header, shift=DOWN * 0.2))

        # ---- reshape (T, N, 8) -> (T*N, 8) ----------------------------------------
        P = 2.4
        st = heatmap(RO["states"][:, :, 1], STATE_COLOR, P, P).move_to([-5.0, 0.3, 0])
        layers = stacked_layers(st, step=0.045)
        st_lab = tensor_label("states", "(128,128,8)", STATE_COLOR, 20).next_to(st, DOWN, buff=0.15)
        self.play(FadeIn(layers), FadeIn(st), FadeIn(st_lab))

        BH, y_bar = 4.3, -0.15
        flat = RO["states"].reshape(T * N, N_STATE)
        flat_small = flat[::64]                                         # every 64th row, display only
        bar = heatmap(flat_small, STATE_COLOR, BH, 0.45).move_to([-2.6, y_bar, 0])
        bar_lab = tensor_label("", "(16384,\\,8)", STATE_COLOR, 20).next_to(bar, DOWN, buff=0.15)
        reshape_code = Text("states.reshape(T*N, -1)", font="Monospace", font_size=18)\
            .next_to(st_lab, DOWN, buff=0.35)
        self.play(FadeIn(reshape_code))
        ghost = Rectangle(width=P, height=P, stroke_color=STATE_COLOR, stroke_width=2).move_to(st)
        self.play(ReplacementTransform(ghost, Rectangle(width=bar.width, height=bar.height,
                                                        stroke_color=STATE_COLOR, stroke_width=2).move_to(bar)),
                  FadeIn(bar, shift=RIGHT * 0.3), run_time=1.5)
        self.play(FadeIn(bar_lab))
        flat_note = Text("time and env dims flattened into one batch dimension", font_size=22, color=TEXT_DIM)\
            .to_edge(DOWN, buff=0.35)
        self.play(FadeIn(flat_note))
        self.wait(1.5)

        # ---- one forward pass --------------------------------------------------
        mlp = VGroup(RoundedRectangle(corner_radius=0.12, width=1.4, height=1.4, stroke_color=WHITE,
                                      stroke_width=2.5, fill_opacity=0.05),
                     Text("model", font_size=22)).move_to([-1.1, y_bar, 0])
        mlp[1].move_to(mlp[0])
        rng = np.random.default_rng(5)
        lg = heatmap(rng.normal(0, 1, (256, 4)), LOGIT_COLOR, BH, 0.4, diverging=True).move_to([0.4, y_bar, 0])
        vv = heatmap(RO["values"].reshape(-1)[::64, None], VALUE_COLOR, BH, 0.16).move_to([1.5, y_bar, 0])
        rr = heatmap(RO["returns"].reshape(-1)[::64, None], POS_COLOR, BH, 0.16,
                     lo=-20, hi=20, diverging=True, neg_color=REWARD_COLOR).move_to([2.7, y_bar, 0])
        aa = heatmap(RO["advantages"].reshape(-1)[::64, None], ADV_COLOR, BH, 0.24,
                     lo=-20, hi=20, diverging=True, neg_color="#9AD1FF").move_to([5.3, y_bar, 0])

        def lab(m, name, shape, color):
            return VGroup(Text(name, font_size=20, color=color), MathTex(shape, font_size=20))\
                .arrange(DOWN, buff=0.06).next_to(m, DOWN, buff=0.15)
        lg_lab = lab(lg, "logits", "(16384,4)", LOGIT_COLOR)
        vv_lab = lab(vv, "V", "(16384,)", VALUE_COLOR)
        rr_lab = lab(rr, "returns R", "(16384,)", RETURN_COLOR)
        aa_lab = lab(aa, "advantage A", "(16384,)", ADV_COLOR)
        for l in (vv_lab, rr_lab, aa_lab):
            l.align_to(lg_lab, UP)

        a1 = Arrow(bar.get_right(), mlp.get_left(), buff=0.1, color=STATE_COLOR)
        a2 = Arrow(mlp.get_right(), lg.get_left(), buff=0.1, color=LOGIT_COLOR)
        fwd = Text("ONE forward pass:  logits, values = model(states)", font_size=22)\
            .to_edge(DOWN, buff=0.35)
        self.play(GrowArrow(a1), FadeIn(mlp), ReplacementTransform(flat_note, fwd))
        self.play(GrowArrow(a2), FadeIn(lg), FadeIn(vv), FadeIn(lg_lab), FadeIn(vv_lab))
        self.wait(1)
        r_note = Text("returns from the backward loop, flattened the same way", font_size=22, color=TEXT_DIM)\
            .to_edge(DOWN, buff=0.35)
        self.play(FadeIn(rr), FadeIn(rr_lab), ReplacementTransform(fwd, r_note))
        self.wait(1)

        a3 = Arrow(rr.get_right(), aa.get_left(), buff=0.12, color=ADV_COLOR)
        a_eq = MathTex(r"A = R - V", font_size=32, color=ADV_COLOR).next_to(a3, UP, buff=0.15)
        detach = Text("advantages = returns - values.detach()", font="Monospace", font_size=17)\
            .next_to(header, DOWN, buff=0.35).to_edge(RIGHT, buff=0.6)
        self.play(GrowArrow(a3), Write(a_eq), FadeIn(aa), FadeIn(aa_lab))
        self.play(FadeIn(detach))
        det_note = Text("detach: A is a fixed weight for the actor, no gradient flows into the critic through it",
                        font_size=20, color=TEXT_DIM).to_edge(DOWN, buff=0.35)
        self.play(ReplacementTransform(r_note, det_note))
        self.wait(2.5)

        # ---- normalization --------------------------------------------------------
        keep = Group(aa, aa_lab)
        self.play(*[FadeOut(m) for m in self.mobjects if m not in (header, aa, aa_lab)])
        self.play(keep.animate.move_to(LEFT * 5.6 + DOWN * 0.1))

        A = RO["advantages"].reshape(-1)
        mu, sd = A.mean(), A.std()
        axis = NumberLine(x_range=[-40, 40, 10], length=7.6, include_numbers=True, font_size=22)\
            .move_to(RIGHT * 1.9 + DOWN * 1.9)
        edges = np.linspace(-40, 40, 41)
        bars = histogram_bars(A, axis, edges, ADV_COLOR, 2.3)
        stats = MathTex(rf"\text{{mean}} = {mu:.2f}, \qquad \text{{std}} = {sd:.2f}", font_size=34)\
            .move_to(RIGHT * 1.9 + UP * 2.3)
        h_title = Text("histogram of all 16,384 advantages", font_size=20, color=TEXT_DIM)\
            .next_to(stats, DOWN, buff=0.15)
        self.play(Create(axis), run_time=0.8)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.03), run_time=1.5)
        self.play(Write(stats), FadeIn(h_title))
        self.wait(1)

        why = VGroup(Text("rewards range from about −100 to +100,", font_size=22),
                     Text("the advantage scale changes from batch to batch", font_size=22))\
            .arrange(DOWN, buff=0.1).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(why))
        self.wait(2)

        norm_eq = MathTex(r"A \;\leftarrow\; \frac{A - \text{mean}(A)}{\text{std}(A) + 10^{-8}}",
                          font_size=34, color=ADV_COLOR).move_to(RIGHT * 1.9 + UP * 1.05)
        self.play(Write(norm_eq))
        self.wait(1)

        axis_n = NumberLine(x_range=[-4, 4, 1], length=7.6, include_numbers=True, font_size=22)\
            .move_to(axis)
        An = (A - mu) / (sd + 1e-8)
        bars_n = histogram_bars(An, axis_n, np.linspace(-4, 4, 41), ADV_COLOR, 2.3)
        stats_n = MathTex(r"\text{mean} = 0, \qquad \text{std} = 1", font_size=34).move_to(stats)
        self.play(ReplacementTransform(axis, axis_n), ReplacementTransform(bars, bars_n),
                  ReplacementTransform(stats, stats_n), run_time=2)
        why2 = VGroup(Text("after normalization: same shape, fixed scale", font_size=22, color=ADV_COLOR),
                      Text("about half the actions are pushed up, half pushed down → stable steps",
                           font_size=20, color=TEXT_DIM)).arrange(DOWN, buff=0.1).to_edge(DOWN, buff=0.3)
        self.play(ReplacementTransform(why, why2))
        self.wait(3)

        nxt = Text("Next: the three loss terms and one optimizer step", font_size=24, color=LOGIT_COLOR)\
            .to_edge(DOWN, buff=0.3)
        self.play(ReplacementTransform(why2, nxt))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# =====================================================================
# Part 6: the three loss terms, gradient clipping, Adam step
# =====================================================================
def simple_bars(values, color, y_scale, origin, spacing=0.75, bw=0.5, chosen=None, nd=2, font=20):
    """Small positive bar chart with value labels and a, a_1 ... names."""
    origin = np.array(origin, dtype=float)
    n = len(values)
    xs = [origin[0] + (i - (n - 1) / 2) * spacing for i in range(n)]
    base = Line([xs[0] - spacing / 2, origin[1], 0], [xs[-1] + spacing / 2, origin[1], 0],
                color=GREY_C, stroke_width=2)
    bars, labs, names = VGroup(), VGroup(), VGroup()
    for i, v in enumerate(values):
        h = max(v * y_scale, 1e-3)
        b = Rectangle(width=bw, height=h, stroke_width=0, fill_color=color, fill_opacity=0.9)\
            .move_to([xs[i], origin[1] + h / 2, 0])
        bars.add(b)
        labs.add(MathTex(f"{v:.{nd}f}", font_size=font).next_to(b, UP, buff=0.06))
        names.add(MathTex(f"a_{i}", font_size=font + 2).move_to([xs[i], origin[1] - 0.22, 0]))
    g = VGroup(base, bars, labs, names)
    g.bars, g.labs, g.names = bars, labs, names
    if chosen is not None:
        g.frame = SurroundingRectangle(bars[chosen], color=WHITE, buff=0.03, stroke_width=3)
        g.add(g.frame)
    return g


def entropy(p):
    p = np.asarray(p, dtype=float)
    return float(-(p * np.log(p)).sum())


class Part06_Losses(LessonPart):
    def construct(self):
        header = section_header("Three loss terms, one update", LOGIT_COLOR)
        self.play(FadeIn(header, shift=DOWN * 0.2))

        # ---- actor card ------------------------------------------------------
        a_title = Text("Actor: policy gradient", font_size=26, color=LOGIT_COLOR)
        a_eq = MathTex(r"L_{actor} = -\,\mathrm{mean}\big(A \cdot \log \pi(a \mid s)\big)", font_size=32)
        VGroup(a_title, a_eq).arrange(DOWN, buff=0.3).move_to(LEFT * 3.5 + UP * 1.85)
        p0 = [0.15, 0.25, 0.40, 0.20]
        p_up = [0.12, 0.38, 0.32, 0.18]
        p_dn = [0.18, 0.12, 0.47, 0.23]
        org = [-3.5, -1.3, 0]
        bars = simple_bars(p0, PROB_COLOR, 3.2, org, chosen=1)
        a_cap = Text("taken action a = 1", font_size=18, color=TEXT_DIM).next_to(bars, DOWN, buff=0.15)

        # ---- critic card -------------------------------------------------------
        c_title = Text("Critic: value regression", font_size=26, color=VALUE_COLOR)
        c_eq = MathTex(r"L_{critic} = \mathrm{mean}\big((R - V(s))^2\big)", font_size=32)
        VGroup(c_title, c_eq).arrange(DOWN, buff=0.3).move_to(RIGHT * 3.5 + UP * 1.85)
        cb = [3.5, -1.3, 0]
        c_base = Line([cb[0] - 1.4, cb[1], 0], [cb[0] + 1.4, cb[1], 0], color=GREY_C, stroke_width=2)
        ys = 2.0
        v_bar = Rectangle(width=0.8, height=0.3 * ys, stroke_width=0, fill_color=VALUE_COLOR, fill_opacity=0.9)\
            .move_to([cb[0] - 0.4, cb[1] + 0.3 * ys / 2, 0])
        r_line = DashedLine([cb[0] - 1.4, cb[1] + 0.8 * ys, 0], [cb[0] + 1.4, cb[1] + 0.8 * ys, 0],
                            color=RETURN_COLOR, stroke_width=3)
        r_lab = MathTex("R", font_size=30, color=RETURN_COLOR).next_to(r_line, RIGHT, buff=0.1)
        v_lab = MathTex("V(s)", font_size=26, color=VALUE_COLOR).next_to(c_base, DOWN, buff=0.12)\
            .set_x(v_bar.get_x())
        c_cap = Text("pull V(s) toward the return R", font_size=18, color=TEXT_DIM)\
            .next_to(v_lab, DOWN, buff=0.1).set_x(cb[0])

        divider = DashedLine(UP * 2.6, DOWN * 2.6, color=GREY_D)
        self.play(FadeIn(a_title), Write(a_eq), FadeIn(c_title), Write(c_eq), Create(divider), run_time=1.5)
        self.play(FadeIn(bars), FadeIn(a_cap), Create(c_base), FadeIn(v_bar), FadeIn(v_lab),
                  Create(r_line), FadeIn(r_lab), FadeIn(c_cap))
        self.wait(1)

        # actor: A > 0 then A < 0 ; critic: V moves toward R
        a_sign = MathTex("A > 0", font_size=32, color=GREEN_C).next_to(bars, UP, buff=0.45).shift(RIGHT * 1.0)
        self.play(FadeIn(a_sign))
        self.play(Transform(bars, simple_bars(p_up, PROB_COLOR, 3.2, org, chosen=1)),
                  v_bar.animate.stretch_to_fit_height(0.55 * ys).move_to([cb[0] - 0.4, cb[1] + 0.55 * ys / 2, 0]),
                  run_time=1.5)
        self.wait(1)
        a_sign2 = MathTex("A < 0", font_size=32, color=RED_C).move_to(a_sign)
        self.play(ReplacementTransform(a_sign, a_sign2),
                  Transform(bars, simple_bars(p_dn, PROB_COLOR, 3.2, org, chosen=1)),
                  v_bar.animate.stretch_to_fit_height(0.75 * ys).move_to([cb[0] - 0.4, cb[1] + 0.75 * ys / 2, 0]),
                  run_time=1.5)
        link = Text("same mechanism as in the first video, now averaged over 16,384 samples",
                    font_size=20, color=TEXT_DIM).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(link))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not header])

        # ---- entropy -----------------------------------------------------------
        e_title = Text("Entropy bonus: keep exploring", font_size=28, color=ADV_COLOR)
        e_eq = MathTex(r"H(\pi) = -\sum_a \pi(a \mid s)\, \log \pi(a \mid s)", font_size=36)
        VGroup(e_title, e_eq).arrange(DOWN, buff=0.3).next_to(header, DOWN, buff=0.4)
        self.play(FadeIn(e_title), Write(e_eq))

        peaked, flat = [0.85, 0.05, 0.05, 0.05], [0.25, 0.25, 0.25, 0.25]
        ch1 = simple_bars(peaked, PROB_COLOR, 1.5, [-3.6, -0.55, 0])
        ch2 = simple_bars(flat, PROB_COLOR, 1.5, [3.6, -0.55, 0])
        h1 = VGroup(Text("confident policy", font_size=22),
                    MathTex(rf"H = {entropy(peaked):.2f}", font_size=32, color=ADV_COLOR))\
            .arrange(DOWN, buff=0.1).next_to(ch1, DOWN, buff=0.2)
        h2 = VGroup(Text("uncertain policy", font_size=22),
                    MathTex(rf"H = {entropy(flat):.2f} = \log 4 \;\text{{(max)}}", font_size=32, color=ADV_COLOR))\
            .arrange(DOWN, buff=0.1).next_to(ch2, DOWN, buff=0.2)
        self.play(FadeIn(ch1), FadeIn(h1))
        self.play(FadeIn(ch2), FadeIn(h2))
        self.wait(1)
        e_note = VGroup(
            Text("the loss contains  − β·H :  minimizing the loss pushes H up", font_size=22),
            Text("→ the policy does not collapse onto one action too early", font_size=22, color=TEXT_DIM),
        ).arrange(DOWN, buff=0.12).move_to([0, -2.25, 0])
        beta = MathTex(r"\beta = 0.01", font_size=34, color=ADV_COLOR).move_to([0, 0.1, 0])
        self.play(FadeIn(e_note, shift=UP * 0.2))
        self.play(Write(beta))
        e_code = code_block(["probs   = torch.exp(log_probs_all)",
                             "entropy = -(probs * log_probs_all).sum(dim=-1).mean()"], font_size=17)\
            .to_edge(DOWN, buff=0.25)
        self.play(FadeIn(e_code))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not header])

        # ---- total loss ----------------------------------------------------
        total = MathTex(r"L", "=", r"L_{actor}", "+", r"0.5\,L_{critic}", "-", r"\beta\,H", font_size=56)
        total[2].set_color(LOGIT_COLOR)
        total[4].set_color(VALUE_COLOR)
        total[6].set_color(ADV_COLOR)
        total.move_to(UP * 1.3)
        braces = VGroup()
        for idx, txt, col in ((2, "better actions", LOGIT_COLOR), (4, "value fit", VALUE_COLOR),
                              (6, "exploration", ADV_COLOR)):
            br = Brace(total[idx], DOWN, color=col)
            braces.add(VGroup(br, Text(txt, font_size=20, color=col).next_to(br, DOWN, buff=0.1)))
        self.play(Write(total), run_time=1.5)
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.1) for b in braces], lag_ratio=0.4))
        l_code = Text("loss = actor_loss + 0.5 * critic_loss - self.entropy_beta * entropy",
                      font="Monospace", font_size=18).next_to(braces, DOWN, buff=0.5)
        one = Text("one shared network  →  one loss  →  one backward pass", font_size=24)\
            .next_to(l_code, DOWN, buff=0.4)
        self.play(FadeIn(l_code))
        self.play(FadeIn(one))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not header])

        # ---- gradient clipping + Adam -----------------------------------------
        g_title = Text("loss.backward(), clip, step", font_size=28)
        g_title.next_to(header, DOWN, buff=0.4)
        self.play(FadeIn(g_title))

        o = LEFT * 3.0 + DOWN * 1.0
        unit = 1.6                                  # scene units per gradient-norm unit
        g_dir = np.array([np.cos(0.6), np.sin(0.6), 0])
        g_norm = 1.8
        g_arrow = Arrow(o, o + g_dir * g_norm * unit, buff=0, color=RED_C, stroke_width=6,
                        max_tip_length_to_length_ratio=0.12)
        g_lab = MathTex(r"g,\;\; \|g\| = 1.8", font_size=30, color=RED_C).next_to(g_arrow.get_end(), RIGHT, buff=0.15)
        circle = DashedVMobject(Circle(radius=0.5 * unit, color=GREY_B), num_dashes=30).move_to(o)
        c_lab = MathTex(r"\|g\| \le 0.5", font_size=26, color=GREY_B).next_to(circle, DOWN, buff=0.12)
        dot = Dot(o, color=WHITE)
        self.play(FadeIn(dot), GrowArrow(g_arrow), FadeIn(g_lab))
        self.play(Create(circle), FadeIn(c_lab))
        clipped = Arrow(o, o + g_dir * 0.5 * unit, buff=0, color=GREEN_C, stroke_width=6,
                        max_tip_length_to_length_ratio=0.3)
        g_lab2 = MathTex(r"g \cdot \tfrac{0.5}{1.8}", font_size=30, color=GREEN_C)\
            .next_to(circle, LEFT, buff=0.25)
        self.play(ReplacementTransform(g_arrow, clipped), ReplacementTransform(g_lab, g_lab2), run_time=1.5)
        clip_txt = VGroup(Text("clip_grad_norm_(params, max_norm=0.5)", font="Monospace", font_size=18),
                          Text("same direction, length capped:", font_size=22),
                          Text("one unlucky batch cannot wreck the policy", font_size=22, color=TEXT_DIM))\
            .arrange(DOWN, buff=0.15, aligned_edge=LEFT).move_to(RIGHT * 3.2 + UP * 0.4)
        self.play(FadeIn(clip_txt, shift=LEFT * 0.2))
        self.wait(2)
        adam = VGroup(Text("optimizer.step()", font="Monospace", font_size=18),
                      MathTex(r"\theta \leftarrow \theta - \text{lr} \cdot \mathrm{Adam}(g), \qquad \text{lr} = 0.002",
                              font_size=30),
                      Text("exactly ONE gradient step per buffer", font_size=22, color=YELLOW))\
            .arrange(DOWN, buff=0.15, aligned_edge=LEFT).next_to(clip_txt, DOWN, buff=0.6).align_to(clip_txt, LEFT)
        self.play(FadeIn(adam, shift=LEFT * 0.2))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# =====================================================================
# Part 7: the training loop
# =====================================================================
class Part07_TrainingLoop(LessonPart):
    def construct(self):
        header = section_header("The training loop", STATE_COLOR)
        self.play(FadeIn(header, shift=DOWN * 0.2))

        def node(lines, color, pos):
            box = RoundedRectangle(corner_radius=0.15, width=3.6, height=1.05, stroke_color=color,
                                   stroke_width=3, fill_color=color, fill_opacity=0.1)
            txt = VGroup(*[Text(l, font_size=20 if i else 24, color=WHITE if i else color)
                           for i, l in enumerate(lines)]).arrange(DOWN, buff=0.08)
            return VGroup(box, txt.move_to(box)).move_to(pos)

        cy = DOWN * 0.35
        n1 = node(["1 · collect", "128 steps in 128 envs"], STATE_COLOR, cy + UP * 1.75)
        n2 = node(["2 · returns", "backward loop, bootstrap"], RETURN_COLOR, cy + RIGHT * 4.3)
        n3 = node(["3 · update", "one batch, one Adam step"], LOGIT_COLOR, cy + DOWN * 1.75)
        n4 = node(["4 · clear", "_reset_buffer()"], GREY_B, cy + LEFT * 4.3)
        nodes = [n1, n2, n3, n4]
        arcs = VGroup(
            ArcBetweenPoints(n1.get_right(), n2.get_top(), angle=-PI / 3),
            ArcBetweenPoints(n2.get_bottom(), n3.get_right(), angle=-PI / 3),
            ArcBetweenPoints(n3.get_left(), n4.get_bottom(), angle=-PI / 3),
            ArcBetweenPoints(n4.get_top(), n1.get_left(), angle=-PI / 3),
        ).set_stroke(GREY_B, 3)
        tips = VGroup(*[Triangle(fill_opacity=1, color=GREY_B).scale(0.09)
                        .move_to(a.get_end()).rotate(angle_of_vector(a.point_from_proportion(1) -
                                                                      a.point_from_proportion(0.95)) - PI / 2)
                        for a in arcs])
        for n in nodes:
            self.play(FadeIn(n, scale=0.95), run_time=0.6)
        self.play(Create(arcs), FadeIn(tips))

        k = ValueTracker(0)
        stats = always_redraw(lambda: VGroup(
            MathTex(rf"\text{{updates}} = {int(k.get_value()):,}".replace(",", "{,}"), font_size=34, color=LOGIT_COLOR),
            MathTex(rf"\text{{loop steps}} = {int(k.get_value()) * 128:,}".replace(",", "{,}"), font_size=28),
            MathTex(rf"\text{{transitions}} = {int(k.get_value()) * 16384:,}".replace(",", "{,}"),
                    font_size=28, color=TEXT_DIM),
        ).arrange(DOWN, buff=0.15).move_to(cy))
        self.add(stats)

        dot = Dot(color=YELLOW, radius=0.1).move_to(arcs[0].get_start())
        self.add(dot)
        for loop in range(3):
            for j, a in enumerate(arcs):
                rt = 0.7 if loop == 0 else 0.35
                self.play(MoveAlongPath(dot, a), Indicate(nodes[(j + 1) % 4][0], color=YELLOW, scale_factor=1.03),
                          run_time=rt, rate_func=linear)
                if j == 1:
                    self.play(k.animate.set_value(loop + 1), run_time=0.2)
        self.wait(0.5)

        whole = Text("your run: 200,000 loop steps", font_size=24, color=YELLOW).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(whole))
        self.play(k.animate.set_value(1562), run_time=3, rate_func=rate_functions.ease_in_out_sine)
        total = Text("= 25.6 M transitions, but only 1,562 gradient steps", font_size=24)\
            .to_edge(DOWN, buff=0.4)
        self.play(ReplacementTransform(whole, total))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not header])

        # ---- on-policy + hyperparameters ------------------------------------------
        onp = VGroup(
            Text("Why throw the buffer away?", font_size=28, color=YELLOW),
            Text("A2C is on-policy: the data must come from the current policy.", font_size=22),
            Text("After one update the policy changed, so old samples are stale.", font_size=22),
            Text("No epochs, no minibatches: simple and robust.", font_size=22, color=GREEN_C),
            Text("(PPO reuses each buffer for several epochs of minibatches,", font_size=20, color=TEXT_DIM),
            Text(" with a clipped objective to make that safe.)", font_size=20, color=TEXT_DIM),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        if onp.width > 7.8:
            onp.scale_to_fit_width(7.8)
        onp.to_edge(LEFT, buff=0.5).set_y(-0.2)

        hp_rows = [("n_envs", "128"), ("n_steps", "128"), ("batch", "16,384"), ("gamma", "0.99"),
                   ("learning_rate", "0.002"), ("entropy_beta", "0.01"), ("max_norm", "0.5"),
                   ("model", "MLP 8→128→128")]
        hp = VGroup(*[VGroup(Text(a, font="Monospace", font_size=18, color=TEXT_DIM),
                             Text(b, font="Monospace", font_size=18)) for a, b in hp_rows])
        for r in hp:
            r[1].next_to(r[0], RIGHT, buff=0.3)
        hp.arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        for r in hp:
            r[1].set_x(hp.get_left()[0] + 2.4 + r[1].width / 2)
        hp_title = Text("hyperparameters", font_size=24, color=STATE_COLOR)
        hp_box = VGroup(hp_title, hp).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        frame = SurroundingRectangle(hp_box, color=GREY_D, buff=0.3, corner_radius=0.1)
        hp_all = VGroup(frame, hp_box).to_edge(RIGHT, buff=0.5).set_y(-0.2)

        self.play(LaggedStart(*[FadeIn(l, shift=RIGHT * 0.2) for l in onp], lag_ratio=0.35), run_time=3)
        self.play(FadeIn(hp_all))
        self.wait(4)
        self.play(*[FadeOut(m) for m in self.mobjects])


# =====================================================================
# Part 8: learning progress and summary
# =====================================================================
# Paste the lines your notebook prints every 1000 steps here
# (format: "n episode_count reward_episode", one per line), e.g.
#   0 0 0.0
#   1000 254 -183.21
#   ...
# Leave empty to use an illustrative curve.
TRAINING_LOG = """
"""


def load_training_curve():
    xs, ys = [], []
    for line in TRAINING_LOG.strip().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[2]))
            except ValueError:
                pass
    if len(xs) >= 5:
        return np.array(xs), np.array(ys), False
    # illustrative curve: from about -200 to about +240
    rng = np.random.default_rng(2)
    xs = np.arange(0, 200001, 1000, dtype=float)
    trend = -200 + 440 / (1 + np.exp(-(xs - 75000) / 16000))
    noise = np.convolve(rng.normal(0, 25, len(xs)), np.ones(5) / 5, mode="same")
    ys = trend + noise
    ys[0] = 0.0
    return xs, ys, True


class Part08_ProgressSummary(LessonPart):
    def construct(self):
        header = section_header("Learning progress", GREEN_C)
        self.play(FadeIn(header, shift=DOWN * 0.2))

        xs, ys, illustrative = load_training_curve()
        x_max = float(np.ceil(xs.max() / 50000) * 50000)
        y_lo = float(min(-300, np.floor(ys.min() / 100) * 100))
        y_hi = float(max(300, np.ceil(ys.max() / 100) * 100))
        ax = Axes(x_range=[0, x_max, 50000], y_range=[y_lo, y_hi, 100], x_length=10, y_length=4.6,
                  axis_config={"color": GREY_B, "include_tip": False},
                  y_axis_config={"include_numbers": True, "font_size": 22})
        ax.move_to(DOWN * 0.35 + RIGHT * 0.3)
        x_nums = VGroup(*[Text(f"{int(v / 1000)}k" if v else "0", font_size=18)
                          .next_to(ax.c2p(v, y_lo), DOWN, buff=0.15)
                          for v in np.arange(0, x_max + 1, 50000)])
        x_lab = Text("loop steps  (each = 128 env transitions)", font_size=20, color=TEXT_DIM)\
            .next_to(x_nums, DOWN, buff=0.15)
        y_lab = Text("episode reward (EMA)", font_size=20, color=TEXT_DIM).rotate(PI / 2)\
            .next_to(ax.y_axis, LEFT, buff=0.6)
        solved = DashedLine(ax.c2p(0, 200), ax.c2p(x_max, 200), color=YELLOW, stroke_width=2.5)
        solved_lab = Text("solved ≈ 200", font_size=20, color=YELLOW).next_to(solved, UP, buff=0.08)\
            .align_to(solved, LEFT).shift(RIGHT * 0.2)
        zero = DashedLine(ax.c2p(0, 0), ax.c2p(x_max, 0), color=GREY_D, stroke_width=1.5)

        self.play(Create(ax), FadeIn(x_nums), FadeIn(x_lab), FadeIn(y_lab))
        self.play(Create(solved), FadeIn(solved_lab), Create(zero))

        curve = ax.plot_line_graph(xs, ys, add_vertex_dots=False, line_color=GREEN_C, stroke_width=3)
        lander = make_lander(0.32)
        t = ValueTracker(0.0)
        lander.add_updater(lambda m: m.move_to(curve["line_graph"].point_from_proportion(
            np.clip(t.get_value(), 0, 1)) + UP * 0.25))
        self.add(lander)
        self.play(Create(curve["line_graph"]), t.animate.set_value(1.0), run_time=5, rate_func=linear)
        lander.clear_updaters()

        above = np.flatnonzero(ys >= 200)
        if len(above):
            xi = xs[above[0]]
            mark = Dot(ax.c2p(xi, ys[above[0]]), color=YELLOW, radius=0.08)
            mark_lab = Text(f"crosses 200 at ≈ {int(xi / 1000)}k steps", font_size=20, color=YELLOW)\
                .next_to(mark, DOWN + RIGHT, buff=0.15)
            self.play(FadeIn(mark, scale=2), FadeIn(mark_lab))
        if illustrative:
            ill = Text("illustrative curve: paste your log into TRAINING_LOG to plot your run",
                       font_size=18, color=TEXT_DIM).to_edge(DOWN, buff=0.2)
            self.play(FadeIn(ill))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ---- summary ------------------------------------------------------------
        s_header = section_header("Summary: one A2C iteration", WHITE)
        self.play(FadeIn(s_header, shift=DOWN * 0.2))
        steps = [
            ("1", STATE_COLOR, "128 envs step together, one batched forward pass per step"),
            ("2", STATE_COLOR, "store 128 steps  →  buffer of 128 × 128 = 16,384 transitions"),
            ("3", RETURN_COLOR, "returns backwards:  R = r + γ·R·(1 − d),  bootstrap from V"),
            ("4", ADV_COLOR, "flatten, one forward pass,  A = R − V,  normalize A"),
            ("5", LOGIT_COLOR, "L = L_actor + 0.5·L_critic − β·H,  clip to 0.5,  one Adam step"),
            ("6", GREY_B, "clear the buffer and repeat"),
        ]
        rows = VGroup()
        for num, col, txt in steps:
            badge = VGroup(Circle(radius=0.22, color=col, fill_opacity=0.2, stroke_width=2.5),
                           Text(num, font_size=22, color=col))
            badge[1].move_to(badge[0])
            rows.add(VGroup(badge, Text(txt, font_size=24)).arrange(RIGHT, buff=0.35))
        rows.arrange(DOWN, buff=0.32, aligned_edge=LEFT).next_to(s_header, DOWN, buff=0.55)
        if rows.width > 12.5:
            rows.scale_to_fit_width(12.5)
        rows.set_x(0)
        for r in rows:
            self.play(FadeIn(r, shift=RIGHT * 0.2), run_time=0.7)
            self.wait(0.6)

        final = VGroup(make_lander(0.45), Text("simple, fast, and it lands.", font_size=28, color=GREEN_C))\
            .arrange(RIGHT, buff=0.3).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(final, shift=UP * 0.2))
        self.wait(4)
        self.play(*[FadeOut(m) for m in self.mobjects])


# =====================================================================
# The video: the ONLY Manim scene in this file -> always one video file
# =====================================================================
PARTS = [Part01_Task, Part02_ParallelEnvs, Part03_RolloutBuffer, Part04_Returns, Part05_BigBatch,
         Part06_Losses, Part07_TrainingLoop, Part08_ProgressSummary]


class A2CTraining(Scene):
    def construct(self):
        # optional: A2C_PARTS="2" renders only part 2 (quick previews)
        sel = os.environ.get("A2C_PARTS", "").strip()
        parts = PARTS if not sel else [PARTS[int(i) - 1] for i in sel.split(",")]
        for part in parts:
            part.construct(self)
            keep = getattr(self, "lunar_shot", None)
            self.clear()
            if keep is not None:          # the screenshot carries over into part 2
                self.add(keep)
            self.wait(0.3)


if __name__ == "__main__":
    # python a2c_training.py  ->  media/videos/1080p30/a2c_training.mp4  (opens when done)
    with tempconfig({"quality": "high_quality", "frame_rate": 30,
                     "output_file": "a2c_training", "preview": True}):
        A2CTraining().render()