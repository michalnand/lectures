"""
A2C explained, one training step, discrete actions.
Manim Community Edition script (tested with manim 0.22).

This file contains ONE Manim scene (A2CLesson) built from 9 parts,
so every command below produces ONE video file.

Full lesson, 1080p 30 fps:
    python a2c_lesson.py
    # or
    manim -pqh --fps 30 a2c_lesson.py

Quick 480p preview of selected parts only (numbers = part order 1..9):
    A2C_PARTS="3,6" manim -pql a2c_lesson.py
"""

from manim import *
import numpy as np
import os


# =====================================================================
# Style: colors follow the matplotlib plots from a2c_principle.ipynb
# =====================================================================
LOGIT_COLOR  = "#4169E1"     # royalblue  -> logits
PROB_COLOR   = "#E03C31"     # red        -> probabilities
VALUE_COLOR  = "#FF8C00"     # darkorange -> critic value V(s)
REWARD_COLOR = "#FF4D4D"     # red dashed -> reward / target
STATE_COLOR  = "#2EC4B6"     # teal       -> state s
CHOSEN_COLOR = WHITE         # frame around the sampled action (dark bg)
HILITE_COLOR = YELLOW        # temporary highlights
TEXT_DIM     = GREY_B

N_STATES  = 4
N_ACTIONS = 3


# =====================================================================
# Toy data, computed once and shared by all scenes, so every number on
# screen is consistent across the whole video.
# Seed 43 gives: sampled action a1 (not the most likely one), A > 0.
# =====================================================================
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def make_toy_data(seed=43, learning_rate=0.5, reward=None):
    """
    One A2C step on a linear toy model (as in a2c_principle.ipynb).
    learning_rate = 0.5 is large on purpose, so the changes are visible on screen.
    reward=None -> reward sampled from the seed, otherwise the given value is used
    (same state, weights and action; used for the A < 0 scene).
    """
    rng = np.random.default_rng(seed)

    # values rounded to 2 decimals, so what we display is exactly what we compute
    s      = np.round(rng.normal(size=(1, N_STATES)), 2)                    # state, (1, 4)
    theta  = np.round(0.5 * rng.normal(size=(N_STATES, N_ACTIONS)), 2)      # actor weights, (4, 3)
    phi    = np.round(0.3 * rng.normal(size=(N_STATES, 1)), 2)              # critic weights, (4, 1)

    logits = (s @ theta)[0]                                                 # (3,)
    probs  = softmax(logits)                                                # (3,)
    action = int(rng.choice(N_ACTIONS, p=probs))
    r_seed = round(float(rng.uniform()), 2)
    reward = r_seed if reward is None else reward

    value  = float((s @ phi)[0, 0])
    adv    = reward - value
    lr     = learning_rate

    # ---- losses ------------------------------------------------------------
    log_pa      = float(np.log(probs[action]))
    actor_loss  = -adv * log_pa
    critic_loss = 0.5 * (value - reward) ** 2

    # ---- actor update: theta <- theta + lr * A * s^T (e_a - pi) ------------
    e_a       = np.zeros(N_ACTIONS); e_a[action] = 1.0
    direction = e_a - probs                                                 # (3,)
    s_norm2   = float((s ** 2).sum())
    theta_new = theta + lr * adv * (s.T @ direction[None, :])
    logits_new = (s @ theta_new)[0]
    coef      = lr * adv * s_norm2                                          # c = alpha A |s|^2
    assert np.allclose(logits_new, logits + coef * direction)              # the key identity
    probs_new = softmax(logits_new)

    # ---- critic update: phi <- phi - lr * (V - r) s^T ----------------------
    phi_new   = phi - lr * (value - reward) * s.T
    value_new = float((s @ phi_new)[0, 0])

    return dict(s=s, theta=theta, phi=phi, logits=logits, probs=probs,
                action=action, reward=reward, value=value, advantage=adv, lr=lr,
                log_pa=log_pa, actor_loss=actor_loss, critic_loss=critic_loss,
                e_a=e_a, direction=direction, s_norm2=s_norm2, coef=coef,
                delta_logits=coef * direction, logits_new=logits_new,
                probs_new=probs_new, value_new=value_new)


DATA = make_toy_data()                    # main example, A > 0
DATA_NEG = make_toy_data(reward=-0.30)    # same step, but a bad reward, A < 0


# =====================================================================
# Helpers
# =====================================================================
def fmt(x, nd=2):
    """Number formatted for LaTeX, negatives wrapped in parentheses."""
    t = f"{x:.{nd}f}"
    return f"({t})" if x < 0 else t


def block(title, formula, color, caption=None, width=3.3, height=1.5):
    """Rounded box with a title, an optional formula and an optional caption."""
    box = RoundedRectangle(corner_radius=0.2, width=width, height=height,
                           stroke_color=color, stroke_width=3,
                           fill_color=color, fill_opacity=0.12)
    parts = [Text(title, font_size=26, color=color)]
    if formula is not None:
        parts.append(MathTex(formula, font_size=36))
    if caption is not None:
        parts.append(Text(caption, font_size=18, color=TEXT_DIM, slant=ITALIC))
    content = VGroup(*parts).arrange(DOWN, buff=0.12).move_to(box)
    return VGroup(box, content)


def make_bars(values, color, y_scale, origin=ORIGIN, spacing=1.6, bar_width=0.8,
              value_color=None, nd=3):
    """
    Simple bar chart supporting negative values.
    Returns a VGroup with attributes: baseline, bars, value_labels, action_labels.
    """
    n = len(values)
    xs = [(i - (n - 1) / 2) * spacing for i in range(n)]

    baseline = Line(LEFT * (spacing * n / 2), RIGHT * (spacing * n / 2),
                    color=GREY_C, stroke_width=2).move_to(origin)
    y0 = baseline.get_y()

    bars, value_labels, action_labels = VGroup(), VGroup(), VGroup()
    for i, (x, v) in enumerate(zip(xs, values)):
        h = max(abs(v) * y_scale, 1e-3)
        bar = Rectangle(width=bar_width, height=h, stroke_width=0,
                        fill_color=color, fill_opacity=0.9)
        cx = origin[0] + x
        if v >= 0:
            bar.move_to([cx, y0 + h / 2, 0])
            lab = MathTex(f"{v:.{nd}f}", font_size=30,
                          color=value_color or WHITE).next_to(bar, UP, buff=0.12)
            act = MathTex(f"a_{i}", font_size=32).move_to([cx, y0 - 0.3, 0])
        else:
            bar.move_to([cx, y0 - h / 2, 0])
            lab = MathTex(f"{v:.{nd}f}", font_size=30,
                          color=value_color or WHITE).next_to(bar, DOWN, buff=0.12)
            act = MathTex(f"a_{i}", font_size=32).move_to([cx, y0 + 0.3, 0])
        bars.add(bar)
        value_labels.add(lab)
        action_labels.add(act)

    chart = VGroup(baseline, bars, value_labels, action_labels)
    chart.baseline, chart.bars = baseline, bars
    chart.value_labels, chart.action_labels = value_labels, action_labels
    return chart


def section_header(text, color):
    return Text(text, font_size=36, color=color).to_edge(UP, buff=0.4)


class LessonPart:
    """One part of the lesson. Not a Manim Scene on its own, so Manim only
    ever finds the single A2CLesson scene and always produces one video."""
    pass


# =====================================================================
# Scene 1: title and big picture
# =====================================================================
class Scene01_Overview(LessonPart):
    def construct(self):
        # ---- title -------------------------------------------------------
        title = Text("Advantage Actor-Critic (A2C)", font_size=52)
        sub = Text("one training step  ·  discrete actions", font_size=28, color=TEXT_DIM)
        VGroup(title, sub).arrange(DOWN, buff=0.3)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(sub, shift=UP * 0.3))
        self.wait(1.5)
        self.play(FadeOut(sub), title.animate.scale(0.6).to_edge(UP, buff=0.35))

        # ---- blocks ------------------------------------------------------
        state_box  = block("State", r"s", STATE_COLOR, "observation",
                           width=2.2).move_to(LEFT * 5.1 + DOWN * 0.2)
        actor_box  = block("Actor", r"\pi_\theta(a \mid s)", LOGIT_COLOR,
                           "what to do?").move_to(LEFT * 0.6 + UP * 1.4)
        critic_box = block("Critic", r"V_\Phi(s)", VALUE_COLOR,
                           "how good is this state?").move_to(LEFT * 0.6 + DOWN * 1.8)
        env_box    = block("Environment", r"r", GREY_B,
                           "returns reward").move_to(RIGHT * 4.6 + UP * 1.4)
        adv_box    = block("Advantage", r"A = r - V(s)", YELLOW,
                           "better than expected?").move_to(RIGHT * 4.6 + DOWN * 1.8)

        # ---- arrows ------------------------------------------------------
        a_state_actor  = Arrow(state_box.get_right(), actor_box.get_left(), buff=0.1, color=STATE_COLOR)
        a_state_critic = Arrow(state_box.get_right(), critic_box.get_left(), buff=0.1, color=STATE_COLOR)

        a_actor_env = Arrow(actor_box.get_right(), env_box.get_left(), buff=0.1, color=LOGIT_COLOR)
        lbl_action = MathTex(r"a \sim \pi_\theta(\cdot \mid s)", font_size=30).next_to(a_actor_env, UP, buff=0.1)

        a_env_adv = Arrow(env_box.get_bottom(), adv_box.get_top(), buff=0.1, color=REWARD_COLOR)
        lbl_r = MathTex("r", font_size=34, color=REWARD_COLOR).next_to(a_env_adv, RIGHT, buff=0.15)

        a_critic_adv = Arrow(critic_box.get_right(), adv_box.get_left(), buff=0.1, color=VALUE_COLOR)
        lbl_v = MathTex("V(s)", font_size=30, color=VALUE_COLOR).next_to(a_critic_adv, UP, buff=0.1)

        a_feedback = CurvedArrow(adv_box.get_left() + UP * 0.3, actor_box.get_bottom() + RIGHT * 0.6,
                                 angle=-PI / 4, color=YELLOW, stroke_width=3)
        lbl_feedback = Text("scales the actor update", font_size=20, color=YELLOW)\
            .move_to(RIGHT * 1.9 + DOWN * 0.15)

        # ---- animate the story ------------------------------------------
        self.play(FadeIn(state_box, shift=RIGHT * 0.3))
        self.wait(0.5)

        self.play(GrowArrow(a_state_actor), FadeIn(actor_box, shift=RIGHT * 0.3))
        self.wait(1)
        self.play(GrowArrow(a_state_critic), FadeIn(critic_box, shift=RIGHT * 0.3))
        self.wait(1)

        self.play(GrowArrow(a_actor_env), Write(lbl_action))
        self.play(FadeIn(env_box, shift=RIGHT * 0.3))
        self.wait(1)

        self.play(GrowArrow(a_env_adv), FadeIn(lbl_r),
                  GrowArrow(a_critic_adv), FadeIn(lbl_v))
        self.play(FadeIn(adv_box, scale=0.9))
        self.wait(1.5)

        self.play(Create(a_feedback), FadeIn(lbl_feedback))
        self.wait(2)

        # ---- closing line -----------------------------------------------
        everything = VGroup(state_box, actor_box, critic_box, env_box, adv_box,
                            a_state_actor, a_state_critic, a_actor_env, lbl_action,
                            a_env_adv, lbl_r, a_critic_adv, lbl_v, a_feedback, lbl_feedback)
        note = Text("Next: one step with real numbers  (4-dim state, 3 actions)",
                    font_size=26, color=TEXT_DIM).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(note, shift=UP * 0.2))
        self.wait(2)
        self.play(FadeOut(everything), FadeOut(note), FadeOut(title))


# =====================================================================
# Scene 2: actor forward pass, logits l = s θ
# =====================================================================
class Scene02_ActorLogits(LessonPart):
    def construct(self):
        s, theta, logits = DATA["s"], DATA["theta"], DATA["logits"]

        header = section_header("1 · Actor forward pass: logits", LOGIT_COLOR)
        eq = MathTex(r"l(s)", r"=", r"s", r"\,\theta", font_size=48)
        eq[0].set_color(LOGIT_COLOR)
        eq[2].set_color(STATE_COLOR)
        eq.next_to(header, DOWN, buff=0.35)

        self.play(FadeIn(header, shift=DOWN * 0.2))
        self.play(Write(eq))
        self.wait(1)

        # ---- matrices ----------------------------------------------------
        cfg = {"num_decimal_places": 2}
        s_mat = DecimalMatrix(s, element_to_mobject_config=cfg, h_buff=1.55)
        s_mat.get_entries().set_color(STATE_COLOR)

        th_mat = DecimalMatrix(theta, element_to_mobject_config=cfg, h_buff=1.55)

        # logits shown as text formatted with Python, same as in the expressions below
        l_mat = Matrix([[f"{x:.3f}" for x in logits]], h_buff=1.7)
        l_mat.get_entries().set_color(LOGIT_COLOR)

        times = MathTex(r"\times", font_size=40)
        equals = MathTex(r"=", font_size=40)
        row = VGroup(s_mat, times, th_mat, equals, l_mat).arrange(RIGHT, buff=0.3)
        row.scale_to_fit_width(13.0).move_to(UP * 0.3)

        s_lab  = MathTex(r"s \;\; (1 \times 4)", font_size=28, color=STATE_COLOR).next_to(s_mat, DOWN, buff=0.25)
        th_lab = MathTex(r"\theta \;\; (4 \times 3)", font_size=28).next_to(th_mat, DOWN, buff=0.25)
        l_lab  = MathTex(r"l \;\; (1 \times 3)", font_size=28, color=LOGIT_COLOR).next_to(l_mat, DOWN, buff=0.25)
        # put the three size labels on a common line
        y_lab = min(m.get_y() for m in (s_lab, th_lab, l_lab))
        for m in (s_lab, th_lab, l_lab):
            m.set_y(y_lab)

        s_note = Text("state features", font_size=20, color=TEXT_DIM).next_to(s_lab, DOWN, buff=0.1)
        th_note = Text("actor weights,\none column per action", font_size=20,
                       color=TEXT_DIM).next_to(th_lab, DOWN, buff=0.1)

        self.play(TransformFromCopy(eq[2], s_mat), FadeIn(s_lab), FadeIn(s_note))
        self.wait(0.5)
        self.play(FadeIn(times), TransformFromCopy(eq[3], th_mat), FadeIn(th_lab), FadeIn(th_note))
        self.wait(0.5)

        # column labels a0 a1 a2 above theta
        cols = th_mat.get_columns()
        col_tags = VGroup(*[MathTex(f"a_{j}", font_size=26, color=LOGIT_COLOR).next_to(cols[j], UP, buff=0.25)
                            for j in range(N_ACTIONS)])
        self.play(FadeIn(col_tags, lag_ratio=0.2))
        self.play(FadeIn(equals), FadeIn(l_mat.get_brackets()), FadeIn(l_lab))
        self.wait(1)

        # ---- dot product, column by column --------------------------------
        s_box = SurroundingRectangle(s_mat.get_entries(), color=HILITE_COLOR, buff=0.08)
        l_entries = l_mat.get_entries()

        expr_prev = None
        col_box = None
        for j in range(N_ACTIONS):
            terms = " + ".join(f"{fmt(s[0, i])} \\cdot {fmt(theta[i, j])}" for i in range(N_STATES))
            expr = MathTex(f"l_{j}", "=", terms, "=", f"{logits[j]:.3f}", font_size=34)
            expr[0].set_color(LOGIT_COLOR)
            expr[4].set_color(LOGIT_COLOR)
            if expr.width > 12.5:
                expr.scale_to_fit_width(12.5)
            expr.to_edge(DOWN, buff=0.8)

            new_col_box = SurroundingRectangle(cols[j], color=HILITE_COLOR, buff=0.1)
            slow = (j == 0)

            if j == 0:
                self.play(Create(s_box), Create(new_col_box))
                explain = Text("each logit = dot product of the state with one column",
                               font_size=24, color=TEXT_DIM).next_to(expr, UP, buff=0.3)
                self.play(FadeIn(explain))
            else:
                self.play(ReplacementTransform(col_box, new_col_box), FadeOut(expr_prev), run_time=0.6)
            col_box = new_col_box

            self.play(Write(expr[:3]), run_time=2.0 if slow else 1.0)
            self.play(Write(expr[3:]), run_time=0.8 if slow else 0.5)
            self.play(TransformFromCopy(expr[4], l_entries[j]), run_time=1.0 if slow else 0.7)
            self.wait(1.0 if slow else 0.4)
            expr_prev = expr

        self.play(FadeOut(expr_prev), FadeOut(explain), FadeOut(col_box), FadeOut(s_box))
        self.wait(0.5)

        # ---- move the matrices up, draw the logits as a bar chart -----------
        self.play(FadeOut(s_note), FadeOut(th_note), run_time=0.5)
        top_group = VGroup(s_mat, times, th_mat, equals, l_mat, col_tags,
                           s_lab, th_lab, l_lab)
        self.play(top_group.animate.scale(0.7).next_to(eq, DOWN, buff=0.3), run_time=1.2)

        y_scale = 1.4 / np.abs(logits).max()
        chart = make_bars(logits, LOGIT_COLOR, y_scale, origin=LEFT * 2.3 + DOWN * 2.3)
        y_title = Text("logits  l", font_size=24, color=LOGIT_COLOR)\
            .next_to(chart.baseline, LEFT, buff=0.4)

        self.play(Create(chart.baseline), FadeIn(y_title), FadeIn(chart.action_labels))
        self.play(*[GrowFromPoint(bar, [bar.get_x(), chart.baseline.get_y(), 0])
                    for bar in chart.bars],
                  run_time=1.2)
        self.play(*[TransformFromCopy(l_entries[j], chart.value_labels[j]) for j in range(N_ACTIONS)],
                  run_time=1.2)
        self.wait(1)

        # ---- takeaway -----------------------------------------------------
        notes = VGroup(
            Text("logits = raw preference scores", font_size=24),
            Text("higher = more preferred", font_size=22, color=TEXT_DIM),
            Text("can be negative, do not sum to 1", font_size=22, color=TEXT_DIM),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(chart, RIGHT, buff=0.7)
        for n in notes:
            self.play(FadeIn(n, shift=LEFT * 0.2), run_time=0.6)
            self.wait(0.6)
        self.wait(1)

        nxt = Text("Next: softmax turns logits into probabilities", font_size=24,
                   color=PROB_COLOR).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(nxt, shift=UP * 0.2))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# =====================================================================
# More helpers (scenes 3 - 9)
# =====================================================================
def vec(values, nd=3):
    """Row vector [x0, x1, x2] in LaTeX."""
    if nd == 0:
        items = [f"{int(round(v))}" for v in values]
    else:
        items = [f"{v:.{nd}f}" for v in values]
    return r"\left[\," + r",\;\; ".join(items) + r"\,\right]"


def value_plot(v, r, origin, y_scale, width=3.6):
    """Critic value bar V(s) plus a dashed reward line and the advantage arrow."""
    origin = np.array(origin, dtype=float)
    y0 = origin[1]
    base = Line(LEFT * width / 2, RIGHT * width / 2, color=GREY_C, stroke_width=2).move_to(origin)

    h = max(abs(v) * y_scale, 1e-3)
    bar = Rectangle(width=1.0, height=h, stroke_width=0, fill_color=VALUE_COLOR, fill_opacity=0.9)
    bar.move_to([origin[0] - 0.6, y0 + np.sign(v) * h / 2, 0])
    v_lab = MathTex(f"V(s) = {v:.3f}", font_size=28, color=VALUE_COLOR)\
        .next_to(bar, UP if v >= 0 else DOWN, buff=0.12)

    y_v = y0 + v * y_scale
    y_r = y0 + r * y_scale
    r_line = DashedLine([origin[0] - width / 2, y_r, 0], [origin[0] + width / 2, y_r, 0],
                        color=REWARD_COLOR, stroke_width=3)
    r_lab = MathTex(f"r = {r:.2f}", font_size=30, color=REWARD_COLOR).next_to(r_line, RIGHT, buff=0.15)

    x_a = origin[0] + 0.6
    a_arrow = Arrow([x_a, y_v, 0], [x_a, y_r, 0], buff=0, color=YELLOW, stroke_width=5,
                    max_tip_length_to_length_ratio=0.2)
    a_lab = MathTex("A", font_size=36, color=YELLOW).next_to(a_arrow, RIGHT, buff=0.15)
    v_guide = DashedLine([bar.get_right()[0], y_v, 0], [x_a, y_v, 0], color=VALUE_COLOR,
                         stroke_width=2, dash_length=0.06)

    g = VGroup(base, bar, v_lab, r_line, r_lab, v_guide, a_arrow, a_lab)
    g.base, g.bar, g.v_lab, g.r_line, g.r_lab = base, bar, v_lab, r_line, r_lab
    g.v_guide, g.a_arrow, g.a_lab = v_guide, a_arrow, a_lab
    return g


def animate_logit_update(scene, D, chart_origin=LEFT * 3.4 + DOWN * 1.2,
                         block_center=RIGHT * 3.3 + UP * 0.35):
    """
    Shared by scene 6 (A > 0) and scene 8 (A < 0):
    numbers for  Δl = αA‖s‖²(e_a − π)  on the right, logits bar chart moving on the left.
    """
    a = D["action"]
    logits, new = D["logits"], D["logits_new"]

    # ---- numbers on the right --------------------------------------------
    lines = VGroup(
        MathTex(r"e_a", "=", vec(D["e_a"], 0), font_size=30),
        MathTex(r"\pi", "=", vec(D["probs"]), font_size=30),
        MathTex(r"e_a - \pi", "=", vec(D["direction"]), font_size=30),
        MathTex(r"c = \alpha A \|s\|^2", "=",
                f"{D['lr']:.1f} \\cdot {fmt(D['advantage'], 3)} \\cdot {D['s_norm2']:.3f}",
                "=", f"{D['coef']:.3f}", font_size=30),
        MathTex(r"\Delta l = c\,(e_a - \pi)", "=", vec(D["delta_logits"]), font_size=30),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
    lines[1][0].set_color(PROB_COLOR)
    lines[3][4].set_color(YELLOW)
    lines[4][0].set_color(LOGIT_COLOR)
    if lines.width > 6.4:
        lines.scale_to_fit_width(6.4)
    lines.move_to(block_center)

    onehot_note = Text(f"one-hot of the chosen action a{a}", font_size=18, color=TEXT_DIM)\
        .next_to(lines[0], RIGHT, buff=0.3)

    # ---- chart on the left ----------------------------------------------
    y_scale = 1.5 / max(np.abs(logits).max(), np.abs(new).max())
    chart = make_bars(logits, LOGIT_COLOR, y_scale, origin=chart_origin, spacing=1.4)
    chart_new = make_bars(new, LOGIT_COLOR, y_scale, origin=chart_origin, spacing=1.4)
    y_title = Text("logits  l", font_size=24, color=LOGIT_COLOR)\
        .next_to(chart.baseline, LEFT, buff=0.3)
    chosen_frame = SurroundingRectangle(chart.action_labels[a], color=CHOSEN_COLOR, buff=0.08)

    scene.play(Create(chart.baseline), FadeIn(y_title), FadeIn(chart.action_labels),
               *[GrowFromPoint(b, [b.get_x(), chart.baseline.get_y(), 0]) for b in chart.bars],
               FadeIn(chart.value_labels), run_time=1.2)
    scene.play(Create(chosen_frame))
    scene.wait(0.5)

    for i, line in enumerate(lines):
        scene.play(Write(line), run_time=1.0)
        if i == 0:
            scene.play(FadeIn(onehot_note), run_time=0.5)
        scene.wait(0.7 if i < 4 else 1.0)

    # ---- arrows showing Δl on each bar ------------------------------------
    y0 = chart.baseline.get_y()
    arrows = VGroup()
    for j in range(N_ACTIONS):
        x = chart.bars[j].get_x() + 0.58
        up = new[j] > logits[j]
        arrows.add(Arrow([x, y0 + logits[j] * y_scale, 0], [x, y0 + new[j] * y_scale, 0],
                         buff=0, color=GREEN_C if up else RED_C, stroke_width=6,
                         max_tip_length_to_length_ratio=0.45, max_stroke_width_to_length_ratio=30))
    scene.play(Indicate(lines[4][2], color=YELLOW), run_time=0.8)
    scene.play(*[GrowArrow(arr) for arr in arrows], run_time=1.2)
    scene.wait(1)

    # ---- move the bars, keep a faded copy of the old ones -------------------
    ghosts = chart.bars.copy().set_fill(opacity=0.25)
    scene.add(ghosts)
    scene.bring_to_front(chart.bars)
    scene.play(*[Transform(chart.bars[j], chart_new.bars[j]) for j in range(N_ACTIONS)],
               *[Transform(chart.value_labels[j], chart_new.value_labels[j]) for j in range(N_ACTIONS)],
               *[Transform(chart.action_labels[j], chart_new.action_labels[j]) for j in range(N_ACTIONS)],
               chosen_frame.animate.move_to(chart_new.action_labels[a]),
               run_time=1.8)
    scene.wait(1)

    return VGroup(lines, onehot_note, chart, y_title, chosen_frame, arrows, ghosts)


def paired_panel(old, new, color, y_scale, origin, names, chosen=None,
                 spacing=1.3, bw=0.42, nd=3):
    """Old (faded) vs new (solid) bars per item, value labels on the new bars."""
    n = len(old)
    origin = np.array(origin, dtype=float)
    y0 = origin[1]
    xs = [origin[0] + (i - (n - 1) / 2) * spacing for i in range(n)]
    base = Line([xs[0] - spacing / 2, y0, 0], [xs[-1] + spacing / 2, y0, 0],
                color=GREY_C, stroke_width=2)
    g = VGroup(base)
    for i in range(n):
        for k, (v, op) in enumerate(((old[i], 0.3), (new[i], 0.95))):
            h = max(abs(v) * y_scale, 1e-3)
            x = xs[i] + (k - 0.5) * (bw + 0.06)
            bar = Rectangle(width=bw, height=h, stroke_width=0, fill_color=color, fill_opacity=op)
            bar.move_to([x, y0 + np.sign(v) * h / 2, 0])
            g.add(bar)
            if k == 1:
                lab = MathTex(f"{v:.{nd}f}", font_size=24)\
                    .next_to(bar, UP if v >= 0 else DOWN, buff=0.08)
                g.add(lab)
                if chosen == i:
                    g.add(SurroundingRectangle(bar, color=CHOSEN_COLOR, buff=0.03, stroke_width=3))
        below = min(old[i], new[i]) >= 0
        g.add(MathTex(names[i], font_size=28).move_to([xs[i], y0 + (-0.3 if below else 0.3), 0]))
    return g


# =====================================================================
# Scene 3: softmax and sampling
# =====================================================================
class Scene03_SoftmaxSampling(LessonPart):
    def construct(self):
        D = DATA
        logits, probs, a = D["logits"], D["probs"], D["action"]

        header = section_header("2 · Softmax and action sampling", PROB_COLOR)
        eq = MathTex(r"\pi_\theta(a \mid s)", "=", r"\frac{e^{l_a}}{\sum_j e^{l_j}}", font_size=44)
        eq[0].set_color(PROB_COLOR)
        eq.next_to(header, DOWN, buff=0.25)
        self.play(FadeIn(header, shift=DOWN * 0.2))
        self.play(Write(eq))
        self.wait(1)

        # ---- logits chart on the left --------------------------------------
        origin = LEFT * 3.0 + DOWN * 0.9
        lchart = make_bars(logits, LOGIT_COLOR, 1.4 / np.abs(logits).max(), origin=origin, spacing=1.4)
        l_title = Text("logits  l", font_size=24, color=LOGIT_COLOR).next_to(lchart.baseline, LEFT, buff=0.3)
        self.play(FadeIn(lchart), FadeIn(l_title))
        self.wait(0.5)

        # ---- table on the right -----------------------------------------
        exps = np.exp(logits)
        Z = exps.sum()

        def cell(txt, color=WHITE):
            return MathTex(txt, font_size=32, color=color)

        corner = cell("x").set_opacity(0)
        rows = [
            [corner] + [cell(f"a_{j}") for j in range(N_ACTIONS)],
            [cell("l_a", LOGIT_COLOR)] + [cell(f"{x:.3f}", LOGIT_COLOR) for x in logits],
            [cell("e^{l_a}")] + [cell(f"{x:.3f}") for x in exps],
            [cell(r"\pi(a \mid s)", PROB_COLOR)] + [cell(f"{x:.3f}", PROB_COLOR) for x in probs],
        ]
        grid = VGroup(*[m for row in rows for m in row])\
            .arrange_in_grid(rows=4, cols=4, buff=(0.5, 0.45), col_alignments="rccc")
        grid.move_to(RIGHT * 3.4 + UP * 0.2)
        R = lambda i: VGroup(*rows[i])

        z_tex = MathTex(r"\textstyle\sum_j e^{l_j}", "=", f"{Z:.3f}", font_size=32)\
            .next_to(grid, DOWN, buff=0.45)

        self.play(FadeIn(R(0)), FadeIn(rows[1][0]))
        self.play(*[TransformFromCopy(lchart.value_labels[j], rows[1][j + 1]) for j in range(N_ACTIONS)])
        self.wait(0.5)

        step1 = Text("1) exponentiate: everything becomes positive", font_size=22, color=TEXT_DIM)\
            .to_edge(DOWN, buff=0.5)
        self.play(FadeIn(step1))
        self.play(FadeIn(rows[2][0]),
                  *[TransformFromCopy(rows[1][j + 1], rows[2][j + 1]) for j in range(N_ACTIONS)], run_time=1.2)
        self.play(Write(z_tex))
        self.wait(1)

        step2 = Text("2) divide by the sum: values add up to 1", font_size=22, color=TEXT_DIM)\
            .to_edge(DOWN, buff=0.5)
        self.play(ReplacementTransform(step1, step2))
        self.play(FadeIn(rows[3][0]),
                  *[TransformFromCopy(rows[2][j + 1], rows[3][j + 1]) for j in range(N_ACTIONS)], run_time=1.2)
        self.wait(1)

        # ---- logits bars become probability bars -----------------------------
        pchart = make_bars(probs, PROB_COLOR, 1.7 / probs.max(), origin=origin, spacing=1.4)
        p_title = Text("probs  π", font_size=24, color=PROB_COLOR).next_to(pchart.baseline, LEFT, buff=0.3)
        self.play(ReplacementTransform(lchart.bars, pchart.bars),
                  ReplacementTransform(lchart.value_labels, pchart.value_labels),
                  ReplacementTransform(lchart.action_labels, pchart.action_labels),
                  ReplacementTransform(l_title, p_title), run_time=1.6)
        sum_tex = MathTex(r"\textstyle\sum_a \pi(a \mid s) = 1", font_size=32, color=PROB_COLOR)\
            .next_to(pchart, UP, buff=0.3)
        step3 = Text("same order as the logits, but now a probability distribution",
                     font_size=22, color=TEXT_DIM).to_edge(DOWN, buff=0.5)
        self.play(Write(sum_tex), ReplacementTransform(step2, step3))
        self.wait(2)

        # ---- sampling ----------------------------------------------------
        self.play(FadeOut(grid), FadeOut(z_tex), FadeOut(step3))
        samp_eq = MathTex(r"a \sim \pi_\theta(\cdot \mid s)", font_size=42).move_to(RIGHT * 3.4 + UP * 1.2)
        self.play(Write(samp_eq))

        strip_w, strip_c = 6.0, RIGHT * 3.4 + DOWN * 0.6
        x_left = strip_c[0] - strip_w / 2
        segs, seg_labs = VGroup(), VGroup()
        cum = 0.0
        for j, p in enumerate(probs):
            seg = Rectangle(width=p * strip_w, height=0.6, stroke_color=BLACK, stroke_width=2,
                            fill_color=PROB_COLOR, fill_opacity=[0.35, 0.6, 0.85][j])
            seg.move_to([x_left + (cum + p / 2) * strip_w, strip_c[1], 0])
            segs.add(seg)
            seg_labs.add(MathTex(f"a_{j}", font_size=30).move_to(seg))
            cum += p
        t0 = MathTex("0", font_size=26).next_to(segs[0], DOWN, buff=0.15).align_to(segs[0], LEFT)
        t1 = MathTex("1", font_size=26).next_to(segs[-1], DOWN, buff=0.15).align_to(segs[-1], RIGHT)

        self.play(*[TransformFromCopy(pchart.bars[j], segs[j]) for j in range(N_ACTIONS)], run_time=1.4)
        self.play(FadeIn(seg_labs), FadeIn(t0), FadeIn(t1))
        how = Text("draw u ~ U(0, 1), pick the segment it lands in", font_size=22, color=TEXT_DIM)\
            .to_edge(DOWN, buff=0.5)
        self.play(FadeIn(how))

        u = probs[:a].sum() + 0.45 * probs[a]          # a point inside the sampled segment
        tracker = ValueTracker(0.0)
        pointer = always_redraw(lambda: Triangle(color=YELLOW, fill_opacity=1).scale(0.13).rotate(PI)
                                .move_to([x_left + tracker.get_value() * strip_w, strip_c[1] + 0.48, 0]))
        self.add(pointer)
        self.play(tracker.animate.set_value(1.0), run_time=0.9, rate_func=linear)
        self.play(tracker.animate.set_value(0.0), run_time=0.7, rate_func=linear)
        self.play(tracker.animate.set_value(u), run_time=1.4, rate_func=rate_functions.ease_out_cubic)
        u_tex = MathTex(f"u = {u:.3f}", font_size=28, color=YELLOW).next_to(pointer, UP, buff=0.1)
        self.play(FadeIn(u_tex))

        seg_frame = SurroundingRectangle(segs[a], color=CHOSEN_COLOR, buff=0.0, stroke_width=4)
        bar_frame = SurroundingRectangle(pchart.bars[a], color=CHOSEN_COLOR, buff=0.04, stroke_width=4)
        result = MathTex(f"a = a_{a}", font_size=44).next_to(segs, DOWN, buff=0.65)
        self.play(Create(seg_frame), Create(bar_frame), Write(result))
        self.wait(1)

        best = int(np.argmax(probs))
        if a != best:
            msg = f"a{best} is the most likely, but we sampled a{a}: this is exploration"
        else:
            msg = f"we sampled a{a}, the most likely action"
        note = Text(msg, font_size=22, color=TEXT_DIM).to_edge(DOWN, buff=0.5)
        self.play(ReplacementTransform(how, note))
        self.wait(2.5)

        nxt = Text("Next: was this action good? Ask the critic.", font_size=24, color=VALUE_COLOR)\
            .to_edge(DOWN, buff=0.5)
        self.play(ReplacementTransform(note, nxt))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# =====================================================================
# Scene 4: critic value and advantage
# =====================================================================
class Scene04_CriticAdvantage(LessonPart):
    def construct(self):
        D = DATA
        s, phi, V, r, A = D["s"], D["phi"], D["value"], D["reward"], D["advantage"]

        header = section_header("3 · Critic and advantage", VALUE_COLOR)
        eq = MathTex(r"V(s)", "=", "s", r"\,\Phi", font_size=48)
        eq[0].set_color(VALUE_COLOR)
        eq[2].set_color(STATE_COLOR)
        eq.next_to(header, DOWN, buff=0.3)
        self.play(FadeIn(header, shift=DOWN * 0.2))
        self.play(Write(eq))
        self.wait(1)

        # ---- s × Φ = V ------------------------------------------------------
        cfg = {"num_decimal_places": 2}
        s_mat = DecimalMatrix(s, element_to_mobject_config=cfg, h_buff=1.55)
        s_mat.get_entries().set_color(STATE_COLOR)
        phi_mat = DecimalMatrix(phi, element_to_mobject_config=cfg)
        v_mat = Matrix([[f"{V:.3f}"]])
        v_mat.get_entries().set_color(VALUE_COLOR)
        row = VGroup(s_mat, MathTex(r"\times"), phi_mat, MathTex("="), v_mat).arrange(RIGHT, buff=0.3)
        row.scale(0.85).move_to(UP * 0.5)
        v_mat.get_entries().set_opacity(0)

        labs = VGroup(
            MathTex(r"s \;\; (1 \times 4)", font_size=28, color=STATE_COLOR).next_to(s_mat, DOWN, buff=0.3),
            MathTex(r"\Phi \;\; (4 \times 1)", font_size=28).next_to(phi_mat, DOWN, buff=0.3),
            MathTex(r"V \;\; (1 \times 1)", font_size=28, color=VALUE_COLOR).next_to(v_mat, DOWN, buff=0.3),
        )
        y_lab = min(m.get_y() for m in labs)
        for m in labs:
            m.set_y(y_lab)

        self.play(TransformFromCopy(eq[2], s_mat), FadeIn(labs[0]))
        self.play(FadeIn(row[1]), TransformFromCopy(eq[3], phi_mat), FadeIn(labs[1]))
        self.play(FadeIn(row[3]), FadeIn(v_mat.get_brackets()), FadeIn(labs[2]))

        terms = " + ".join(f"{fmt(s[0, i])} \\cdot {fmt(phi[i, 0])}" for i in range(N_STATES))
        expr = MathTex("V(s)", "=", terms, "=", f"{V:.3f}", font_size=34)
        expr[0].set_color(VALUE_COLOR)
        expr[4].set_color(VALUE_COLOR)
        if expr.width > 12.5:
            expr.scale_to_fit_width(12.5)
        expr.to_edge(DOWN, buff=0.9)

        boxes = VGroup(SurroundingRectangle(s_mat.get_entries(), color=HILITE_COLOR, buff=0.08),
                       SurroundingRectangle(phi_mat.get_entries(), color=HILITE_COLOR, buff=0.08))
        self.play(Create(boxes))
        self.play(Write(expr[:3]), run_time=1.5)
        self.play(Write(expr[3:]))
        v_entry = v_mat.get_entries()[0]
        v_new = v_entry.copy().set_opacity(1)
        self.play(TransformFromCopy(expr[4], v_new))
        meaning = Text("a single number: how much reward the critic expects from state s",
                       font_size=22, color=TEXT_DIM).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(meaning))
        self.wait(2)

        self.play(*[FadeOut(m) for m in (row, labs, boxes, expr, meaning, v_new)])

        # ---- value bar, reward line, advantage arrow ---------------------------
        vp = value_plot(V, r, origin=LEFT * 3.4 + DOWN * 2.6, y_scale=3.6, width=4.4)
        self.play(Create(vp.base), GrowFromPoint(vp.bar, [vp.bar.get_x(), vp.base.get_y(), 0]),
                  FadeIn(vp.v_lab))
        self.wait(0.5)

        r_txt = Text("the environment returns a reward", font_size=22, color=REWARD_COLOR)\
            .move_to(RIGHT * 3.2 + UP * 1.3)
        self.play(Create(vp.r_line), FadeIn(vp.r_lab), FadeIn(r_txt))
        self.wait(1)

        adv_eq = MathTex("A", "=", "r", "-", "V(s)", font_size=44).move_to(RIGHT * 3.2 + UP * 0.4)
        adv_eq[0].set_color(YELLOW)
        adv_eq[2].set_color(REWARD_COLOR)
        adv_eq[4].set_color(VALUE_COLOR)
        adv_num = MathTex("A", "=", f"{r:.2f}", "-", f"{V:.3f}", "=", f"{A:.3f}", font_size=40)\
            .next_to(adv_eq, DOWN, buff=0.35)
        adv_num[0].set_color(YELLOW)
        adv_num[2].set_color(REWARD_COLOR)
        adv_num[4].set_color(VALUE_COLOR)
        adv_num[6].set_color(YELLOW)

        self.play(Write(adv_eq))
        self.play(Create(vp.v_guide), GrowArrow(vp.a_arrow), FadeIn(vp.a_lab))
        self.play(TransformFromCopy(adv_eq, adv_num))
        self.wait(1)

        a = D["action"]
        if A > 0:
            verdict = VGroup(Text("A > 0: better than the critic expected", font_size=24, color=GREEN_C),
                             Text(f"so make action a{a} more likely", font_size=24))
        else:
            verdict = VGroup(Text("A < 0: worse than the critic expected", font_size=24, color=RED_C),
                             Text(f"so make action a{a} less likely", font_size=24))
        verdict.arrange(DOWN, buff=0.15).next_to(adv_num, DOWN, buff=0.5)
        self.play(FadeIn(verdict[0], shift=UP * 0.2))
        self.play(FadeIn(verdict[1], shift=UP * 0.2))
        self.wait(1)

        simpl = Text("simplified: one step, r is the target.  In a2c_agent.py the n-step return R replaces r.",
                     font_size=18, color=TEXT_DIM).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(simpl))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# =====================================================================
# Scene 5: loss functions
# =====================================================================
class Scene05_Losses(LessonPart):
    def construct(self):
        D = DATA
        V, r, A, a = D["value"], D["reward"], D["advantage"], D["action"]
        pa = D["probs"][a]

        header = section_header("4 · Loss functions", WHITE)
        self.play(FadeIn(header, shift=DOWN * 0.2))

        # ---- critic column -----------------------------------------------
        c_title = Text("Critic: mean squared error", font_size=28, color=VALUE_COLOR)
        c_eq = MathTex(r"L_{critic}", "=", r"\tfrac{1}{2}\,\big(V(s) - r\big)^2", font_size=40)
        c_num = MathTex("=", rf"\tfrac{{1}}{{2}}\,({V:.3f} - {r:.2f})^2", "=", f"{D['critic_loss']:.3f}",
                        font_size=36)
        c_note = Text("pulls V(s) toward the reward", font_size=22, color=TEXT_DIM)
        c_col = VGroup(c_title, c_eq, c_num, c_note).arrange(DOWN, buff=0.4)
        c_num.align_to(c_eq[1], LEFT)

        # ---- actor column ---------------------------------------------------
        a_title = Text("Actor: policy gradient", font_size=28, color=LOGIT_COLOR)
        a_eq = MathTex(r"L_{actor}", "=", r"-A \, \log \pi_\theta(a \mid s)", font_size=40)
        a_num = MathTex("=", rf"-{fmt(A, 3)} \cdot \log {pa:.3f}", "=", f"{D['actor_loss']:.3f}",
                        font_size=36)
        a_note = Text("A is a constant here (detached)", font_size=22, color=TEXT_DIM)
        a_col = VGroup(a_title, a_eq, a_num, a_note).arrange(DOWN, buff=0.4)
        a_num.align_to(a_eq[1], LEFT)

        cols = VGroup(c_col, a_col).arrange(RIGHT, buff=1.0, aligned_edge=UP)
        if cols.width > 13.2:
            cols.scale_to_fit_width(13.2)
        cols.next_to(header, DOWN, buff=0.6)
        divider = DashedLine(UP * 0.5, DOWN * 0.5, color=GREY_D)\
            .stretch_to_fit_height(cols.height).move_to(
                [(c_col.get_right()[0] + a_col.get_left()[0]) / 2, cols.get_y(), 0])

        for col in (c_col, a_col):
            self.play(FadeIn(col[0], shift=DOWN * 0.2))
            self.play(Write(col[1]))
            self.wait(0.5)
            self.play(Write(col[2]), run_time=1.5)
            self.play(FadeIn(col[3]))
            self.wait(1.5)
        self.play(Create(divider))

        # ---- intuition ------------------------------------------------------------
        if A > 0:
            intu = MathTex(r"A > 0:\;\; \text{minimizing } L_{actor} \text{ raises } \log \pi_\theta(a_%d \mid s)" % a,
                           font_size=34)
        else:
            intu = MathTex(r"A < 0:\;\; \text{minimizing } L_{actor} \text{ lowers } \log \pi_\theta(a_%d \mid s)" % a,
                           font_size=34)
        intu.next_to(cols, DOWN, buff=0.6)
        box = SurroundingRectangle(intu, color=YELLOW, buff=0.15)
        self.play(Write(intu), Create(box))
        self.wait(2)

        # ---- link to a2c_agent.py ---------------------------------------------
        full = MathTex(r"\texttt{a2c\_agent.py:}\quad L", "=", r"L_{actor}", "+",
                       r"0.5\,\big(R - V(s)\big)^2", "-", r"\beta\, H(\pi)", font_size=34)
        full[2].set_color(LOGIT_COLOR)
        full[4].set_color(VALUE_COLOR)
        full.to_edge(DOWN, buff=0.95)
        full_note = Text("R = n-step return,  averaged over the batch,  H = entropy bonus to keep exploring",
                         font_size=20, color=TEXT_DIM).next_to(full, DOWN, buff=0.2)
        self.play(FadeIn(full, shift=UP * 0.2))
        self.play(FadeIn(full_note))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# =====================================================================
# Scene 6: how the gradient moves the logits (core of the lesson)
# =====================================================================
class Scene06_LogitUpdate(LessonPart):
    def construct(self):
        D = DATA
        header = section_header("5 · How the update moves the logits", LOGIT_COLOR)
        self.play(FadeIn(header, shift=DOWN * 0.2))

        g1 = MathTex(r"\nabla_\theta \log \pi_\theta(a \mid s)", "=", r"s^T (e_a - \pi)", font_size=40)
        g1_note = Text("e_a = one-hot vector of the chosen action", font_size=22, color=TEXT_DIM)
        g2 = MathTex(r"\theta_{new}", "=", r"\theta + \alpha\, A\, s^T (e_a - \pi)", font_size=40)
        g2_note = Text("gradient step on L_actor, scaled by the advantage", font_size=22, color=TEXT_DIM)
        g3 = MathTex(r"l_{new} = s\,\theta_{new}", "=", r"l + \alpha\, A\, s\, s^T (e_a - \pi)", font_size=40)
        g3_note = MathTex(r"s\, s^T = \|s\|^2 \;\; \text{is just a number (s is a row vector)}",
                          font_size=30, color=TEXT_DIM)
        g4 = MathTex(r"\Delta l", "=", r"\alpha\, A\, \|s\|^2", r"\,(e_a - \pi)", font_size=48)
        g4[0].set_color(LOGIT_COLOR)
        g4[2].set_color(YELLOW)

        block = VGroup(g1, g1_note, g2, g2_note, g3, g3_note, g4).arrange(DOWN, buff=0.22)
        for m in (g1_note, g2_note, g3_note):
            m.shift(UP * 0.05)
        block.next_to(header, DOWN, buff=0.4)
        if block.height > 6.4:
            block.scale_to_fit_height(6.4).next_to(header, DOWN, buff=0.4)

        for eq_, note in ((g1, g1_note), (g2, g2_note), (g3, g3_note)):
            self.play(Write(eq_), run_time=1.4)
            self.play(FadeIn(note))
            self.wait(1.5)
        self.play(TransformFromCopy(g3, g4), run_time=1.5)
        box = SurroundingRectangle(g4, color=YELLOW, buff=0.15)
        self.play(Create(box))
        self.wait(1)

        brace_c = Brace(g4[2], DOWN, color=YELLOW)
        brace_c_t = MathTex("c", font_size=34, color=YELLOW).next_to(brace_c, DOWN, buff=0.1)
        brace_d = Brace(g4[3], DOWN, color=GREY_B)
        brace_d_t = Text("direction", font_size=22, color=GREY_B).next_to(brace_d, DOWN, buff=0.1)
        self.play(FadeOut(VGroup(g1, g1_note, g2, g2_note, g3, g3_note)))
        top = VGroup(g4, box)
        self.play(top.animate.next_to(header, DOWN, buff=0.3))
        brace_c = Brace(g4[2], DOWN, color=YELLOW, buff=0.25)
        brace_c_t.next_to(brace_c, DOWN, buff=0.05)
        brace_d = Brace(g4[3], DOWN, color=GREY_B, buff=0.25)
        brace_d_t.next_to(brace_d, DOWN, buff=0.05)
        self.play(GrowFromCenter(brace_c), FadeIn(brace_c_t), GrowFromCenter(brace_d), FadeIn(brace_d_t))
        self.wait(1.5)
        self.play(FadeOut(VGroup(brace_c, brace_c_t, brace_d, brace_d_t)))

        stuff = animate_logit_update(self, D)

        a = D["action"]
        rule = MathTex(r"\Delta l_{a} = c\,(1 - \pi_{a})", r"\;\;\text{(chosen)}", r"\qquad",
                       r"\Delta l_j = -\,c\,\pi_j", r"\;\;\text{(others)}", font_size=32)
        rule[1].set_color(TEXT_DIM)
        rule[4].set_color(TEXT_DIM)
        rule.to_edge(DOWN, buff=0.75)
        rule_note = Text("A > 0: chosen logit goes up, the others go down, more likely competitors are pushed harder",
                         font_size=19, color=TEXT_DIM).next_to(rule, DOWN, buff=0.15)
        self.play(Write(rule))
        self.play(FadeIn(rule_note))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# =====================================================================
# Scene 7: before vs after
# =====================================================================
class Scene07_BeforeAfter(LessonPart):
    def construct(self):
        D = DATA
        a = D["action"]
        header = section_header("6 · Before vs after one update", WHITE)
        legend = Text("faded = before,   solid = after", font_size=22, color=TEXT_DIM)\
            .next_to(header, DOWN, buff=0.2)
        self.play(FadeIn(header, shift=DOWN * 0.2), FadeIn(legend))

        names = [f"a_{j}" for j in range(N_ACTIONS)]
        y_base = -1.0

        l_all = np.concatenate([D["logits"], D["logits_new"]])
        p_logits = paired_panel(D["logits"], D["logits_new"], LOGIT_COLOR, 1.3 / np.abs(l_all).max(),
                                [-4.7, y_base, 0], names, chosen=a)
        p_probs = paired_panel(D["probs"], D["probs_new"], PROB_COLOR,
                               2.4 / max(D["probs"].max(), D["probs_new"].max()),
                               [0.0, y_base, 0], names, chosen=a)

        # value panel: old and new V(s) with the reward line
        V, Vn, r = D["value"], D["value_new"], D["reward"]
        ys = 2.4 / max(abs(V), abs(Vn), abs(r))
        vx = 4.7
        v_base = Line([vx - 1.3, y_base, 0], [vx + 1.3, y_base, 0], color=GREY_C, stroke_width=2)
        v_bars, v_labs = VGroup(), VGroup()
        for k, (v, op, nm) in enumerate(((V, 0.3, "old"), (Vn, 0.95, "new"))):
            h = max(abs(v) * ys, 1e-3)
            b = Rectangle(width=0.8, height=h, stroke_width=0, fill_color=VALUE_COLOR, fill_opacity=op)
            b.move_to([vx + (k - 0.5) * 1.0, y_base + np.sign(v) * h / 2, 0])
            v_bars.add(b)
            v_labs.add(VGroup(MathTex(f"{v:.3f}", font_size=24), Text(nm, font_size=18, color=TEXT_DIM))
                       .arrange(DOWN, buff=0.06)
                       .next_to(b, DOWN if v >= 0 else UP, buff=0.12))
        r_line = DashedLine([vx - 1.3, y_base + r * ys, 0], [vx + 1.3, y_base + r * ys, 0],
                            color=REWARD_COLOR, stroke_width=3)
        r_lab = MathTex("r", font_size=30, color=REWARD_COLOR).next_to(r_line, RIGHT, buff=0.1)
        p_value = VGroup(v_base, v_bars, v_labs, r_line, r_lab)

        titles = VGroup(
            Text("logits  l", font_size=26, color=LOGIT_COLOR).move_to([-4.7, 1.9, 0]),
            Text("probs  π", font_size=26, color=PROB_COLOR).move_to([0.0, 1.9, 0]),
            Text("critic  V(s)", font_size=26, color=VALUE_COLOR).move_to([4.7, 1.9, 0]),
        )

        for title, panel in zip(titles, (p_logits, p_probs, p_value)):
            self.play(FadeIn(title), FadeIn(panel, lag_ratio=0.05), run_time=1.3)
            self.wait(1.2)

        summary = VGroup(
            MathTex(rf"\pi(a_{a} \mid s):\; {D['probs'][a]:.3f} \;\rightarrow\; {D['probs_new'][a]:.3f}",
                    font_size=32, color=PROB_COLOR),
            MathTex(rf"V(s):\; {V:.3f} \;\rightarrow\; {Vn:.3f} \quad (r = {r:.2f})",
                    font_size=32, color=VALUE_COLOR),
        ).arrange(RIGHT, buff=1.2).to_edge(DOWN, buff=1.05)
        self.play(FadeIn(summary, shift=UP * 0.2))
        self.wait(1.5)

        take = Text("The advantage decides the direction, the gradient decides which logits move.",
                    font_size=24).to_edge(DOWN, buff=0.4)
        self.play(Write(take), run_time=2)
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# =====================================================================
# Scene 8: the same step with A < 0
# =====================================================================
class Scene08_NegativeAdvantage(LessonPart):
    def construct(self):
        D = DATA_NEG
        V, r, A, a = D["value"], D["reward"], D["advantage"], D["action"]

        header = section_header("What if A < 0 ?", YELLOW)
        sub = Text(f"same state, same action a{a}, but a bad reward  r = {r:.2f}",
                   font_size=24, color=TEXT_DIM).next_to(header, DOWN, buff=0.25)
        self.play(FadeIn(header, shift=DOWN * 0.2), FadeIn(sub))

        vp = value_plot(V, r, origin=LEFT * 3.4 + DOWN * 0.5, y_scale=2.4)
        self.play(Create(vp.base), GrowFromPoint(vp.bar, [vp.bar.get_x(), vp.base.get_y(), 0]),
                  FadeIn(vp.v_lab))
        self.play(Create(vp.r_line), FadeIn(vp.r_lab))
        self.play(Create(vp.v_guide), GrowArrow(vp.a_arrow), FadeIn(vp.a_lab))

        adv_num = MathTex("A", "=", f"{r:.2f}", "-", f"{V:.3f}", "=", f"{A:.3f}", font_size=40)\
            .move_to(RIGHT * 3.2 + UP * 0.6)
        adv_num[0].set_color(YELLOW)
        adv_num[2].set_color(REWARD_COLOR)
        adv_num[4].set_color(VALUE_COLOR)
        adv_num[6].set_color(YELLOW)
        verdict = VGroup(Text("A < 0: worse than the critic expected", font_size=24, color=RED_C),
                         Text(f"so make action a{a} less likely", font_size=24))\
            .arrange(DOWN, buff=0.15).next_to(adv_num, DOWN, buff=0.5)
        self.play(Write(adv_num))
        self.play(FadeIn(verdict, shift=UP * 0.2))
        self.wait(2.5)
        self.play(FadeOut(VGroup(vp, adv_num, verdict, sub)))

        key = MathTex(r"\Delta l", "=", r"\alpha\, A\, \|s\|^2", r"\,(e_a - \pi)", font_size=44)
        key[0].set_color(LOGIT_COLOR)
        key[2].set_color(YELLOW)
        key.next_to(header, DOWN, buff=0.3)
        key_box = SurroundingRectangle(key, color=YELLOW, buff=0.12)
        self.play(Write(key), Create(key_box))
        self.wait(0.5)

        animate_logit_update(self, D)

        take = VGroup(
            Text("same direction (e_a − π), the negative A flips every arrow", font_size=24),
            MathTex(rf"\pi(a_{a} \mid s):\; {D['probs'][a]:.3f} \;\rightarrow\; {D['probs_new'][a]:.3f}",
                    font_size=32, color=PROB_COLOR),
        ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(take, shift=UP * 0.2))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# =====================================================================
# Scene 9: summary
# =====================================================================
class Scene09_Summary(LessonPart):
    def construct(self):
        header = section_header("Summary: one A2C step", WHITE)
        self.play(FadeIn(header, shift=DOWN * 0.2))

        def row(tag, color, tex):
            t = Text(tag, font_size=24, color=color)
            m = MathTex(tex, font_size=36)
            return VGroup(t, m)

        rows = VGroup(
            row("actor", LOGIT_COLOR, r"l = s\,\theta, \quad \pi = \mathrm{softmax}(l), \quad a \sim \pi"),
            row("critic", VALUE_COLOR, r"V(s) = s\,\Phi"),
            row("advantage", YELLOW, r"A = r - V(s)"),
            row("losses", WHITE, r"L_{actor} = -A \log \pi_\theta(a \mid s), \quad L_{critic} = \tfrac{1}{2}(V(s) - r)^2"),
            row("logit change", LOGIT_COLOR, r"\Delta l = \alpha A \|s\|^2 (e_a - \pi)"),
        )
        tag_w = max(r[0].width for r in rows)
        for r_ in rows:
            r_[1].next_to(r_[0], RIGHT, buff=0.5 + tag_w - r_[0].width)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.42).next_to(header, DOWN, buff=0.6)

        for r_ in rows:
            self.play(FadeIn(r_[0]), Write(r_[1]), run_time=1.2)
            self.wait(0.8)

        agent = Text("a2c_agent.py does the same with a neural network, 128 envs × n_steps in one batch,\n"
                     "n-step returns R instead of r, normalized advantages and an entropy bonus.",
                     font_size=21, color=TEXT_DIM, line_spacing=1.2).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(agent, shift=UP * 0.2))
        self.wait(4)
        self.play(*[FadeOut(m) for m in self.mobjects])


# =====================================================================
# The lesson: the ONLY Manim scene in this file -> always one video file
# =====================================================================
LESSON = [Scene01_Overview, Scene02_ActorLogits, Scene03_SoftmaxSampling,
          Scene04_CriticAdvantage, Scene05_Losses, Scene06_LogitUpdate,
          Scene07_BeforeAfter, Scene08_NegativeAdvantage, Scene09_Summary]


class A2CLesson(Scene):
    def construct(self):
        # optional: A2C_PARTS="3,6" renders only parts 3 and 6 (quick previews)
        sel = os.environ.get("A2C_PARTS", "").strip()
        parts = LESSON if not sel else [LESSON[int(i) - 1] for i in sel.split(",")]
        for part in parts:
            part.construct(self)     # every part ends by fading out its objects
            self.clear()
            self.wait(0.4)


if __name__ == "__main__":
    # python a2c_lesson.py  ->  media/videos/1080p30/a2c_lesson.mp4  (opens when done)
    with tempconfig({"quality": "high_quality", "frame_rate": 30,
                     "output_file": "a2c_lesson", "preview": True}):
        A2CLesson().render()
