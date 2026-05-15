"""Generate status_denoisers.html from per-run metrics.csv + figures."""
from __future__ import annotations

import csv
from pathlib import Path

RUNS_DIR  = Path("~/Runs/remote/WarpX/2d_planar_pinch").expanduser()
DOCS_DIR  = Path("~/Documents/Work_Docs/2026_04_ML_ImplicitPIC").expanduser()
FIG_DIR   = "figures/denoisers"

CHANNELS = [
    "jx", "jy", "jz",
    "num_electrons",
    "px_electrons", "py_electrons", "pz_electrons",
    "enex_electrons", "eney_electrons", "enez_electrons",
    "num_deuterium",
    "px_deuterium", "py_deuterium", "pz_deuterium",
    "enex_deuterium", "eney_deuterium", "enez_deuterium",
]

DERIVED_CHANNELS = [
    "Pxx_electrons", "Pyy_electrons", "Pzz_electrons",
    "Tx_electrons",  "Ty_electrons",  "Tz_electrons",
    "Pxx_deuterium", "Pyy_deuterium", "Pzz_deuterium",
    "Tx_deuterium",  "Ty_deuterium",  "Tz_deuterium",
]

TIERS = ["ppc0036", "ppc0064", "ppc0144", "ppc0256"]
STEPS = ["011025", "054900", "076950", "105300"]
PHASE_LABEL = {
    "011025": "Ramp-up (step 011025)",
    "054900": "Peak compression (step 054900)",
    "076950": "Post-peak (step 076950)",
    "105300": "Rebound (step 105300)",
}
TIER_LABEL = {
    "ppc0036": "N<sub>ppc</sub> = 36 (highest noise)",
    "ppc0064": "N<sub>ppc</sub> = 64",
    "ppc0144": "N<sub>ppc</sub> = 144",
    "ppc0256": "N<sub>ppc</sub> = 256 (lowest noise)",
}


VARIANTS = [
    {
        "id":      "charbonnier",
        "anchor":  "charbonnier",
        "section": "4.3.2.1",
        "title":   "Global U-Net / Charbonnier",
        "run_dir": "globalcnn_charbonnier",
        "loss_intro":
            "Reconstruction loss: per-channel-weighted Charbonnier "
            "(<em>ρ</em>(<em>x</em>) = √(<em>x</em>² + ε²) − ε with "
            "ε = 10⁻³). Spectral, lowpass charge-continuity, and "
            "closure-consistency terms active at the base-config "
            "weights. Gradient-of-error term off.",
    },
    {
        "id":      "charbonnier_grad",
        "anchor":  "charbonnier-grad",
        "section": "4.3.2.2",
        "title":   "Global U-Net / Charbonnier + gradient-of-error",
        "run_dir": "globalcnn_charbonnier_grad",
        "loss_intro":
            "Same as the Charbonnier variant above plus a "
            "gradient-of-error term (L<sub>2</sub> on the spatial "
            "gradient of the residual). Sharpens edges; at the cost "
            "of a slightly higher overall val loss because the new "
            "term enters the total.",
    },
    {
        "id":      "localcnn",
        "anchor":  "localcnn",
        "section": "4.3.2.3",
        "title":   "Local CNN / Charbonnier",
        "run_dir": "localcnn_charbonnier",
        "loss_intro":
            "Local-physics denoiser: a small CNN looks at a 7&times;7 "
            "input patch and emits a single centre-cell prediction. "
            "Reconstruction loss: per-channel-weighted Charbonnier "
            "(<em>&rho;</em>(<em>x</em>) = &radic;(<em>x</em>&sup2; + "
            "&epsilon;&sup2;) &minus; &epsilon; with "
            "&epsilon; = 10&#8315;&sup3;). Closure-consistency term "
            "active at weight 1.0; cross-channel spatial-derivative "
            "terms (spectral, gradient, charge continuity) are off "
            "because a 1&times;1 readout cannot evaluate them. See "
            "<a href=\"approach_localParticleDenoiser.html\">&sect;3.2 "
            "localParticleDenoiser</a> for the architecture.",
    },
    {
        "id":      "hlocalcnn",
        "anchor":  "hlocalcnn",
        "section": "4.3.2.4",
        "title":   "Hierarchical-readout Local CNN / Charbonnier (degenerate)",
        "run_dir": "hlocalcnn_charbonnier",
        "loss_intro":
            "Same 7&times;7-input / centre-cell-output contract as the "
            "Local CNN, but with intermediate-scale readouts at 3&times;3, "
            "5&times;5, and 7&times;7. Each non-1&times;1 readout owns a "
            "learnable per-cell damping mask, and there are 17 learnable "
            "per-channel loss-contribution weights at each scale. The "
            "intent (task #89) was to let the optimizer discover which "
            "scales carry which physical constraints. <strong>Result: the "
            "model trains to a degenerate solution.</strong> The optimizer "
            "drove every learnable multiplier toward "
            "softplus<sup>&minus;1</sup>(0), collapsing the total loss "
            "to ~10<sup>&minus;7</sup> while the predictions stayed at "
            "the zero-init passthrough baseline; per-channel RMSE at "
            "the best checkpoint matches the noisy input. The bug is in "
            "the loss-multiplier parameterisation, not the architecture; "
            "the planned fix is to either fix the multipliers at 1.0 or "
            "constrain them with a softmax across scales so they cannot "
            "collapse. See "
            "<a href=\"approach_hLocalParticleDenoiser.html\">&sect;3.3 "
            "hLocalParticleDenoiser</a> for the architecture.",
    },
    {
        "id":      "flowunet",
        "anchor":  "flowunet",
        "section": "4.3.2.5",
        "title":   "Flow-matching U-Net (linear interpolant)",
        "run_dir": "flowunet_linear",
        "loss_intro":
            "Generative-style denoiser. The U-Net (with FiLM-modulated "
            "time conditioning) learns the velocity field of a linear "
            "interpolant between the noisy low-ppc snapshot (<em>x</em>"
            "<sub>0</sub>) and the ppc-1024 ground truth (<em>x</em>"
            "<sub>1</sub>); at predict time we integrate the resulting "
            "ODE with a 5-step Euler sampler. Loss = pointwise MSE "
            "between predicted velocity and the true linear-interpolant "
            "velocity (<em>x</em><sub>1</sub> − <em>x</em><sub>0</sub>); "
            "no auxiliary cross-channel losses on the integrated output. "
            "See <a href=\"approach_flowParticleDenoiser.html\">§3.6 "
            "flowParticleDenoiser</a> for the architecture.",
    },
]


def parsePerChannelRmse(s: str) -> list[float]:
    return [float(x) for x in s.split(";")] if s else []


def extract(metrics_path: Path) -> dict:
    val_rows = []
    with metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for row in reader:
            if row["split"] != "val":
                continue
            val_rows.append({c: row.get(c, "") for c in cols})
    best = min(val_rows, key=lambda r: float(r["loss"]))
    return {
        "best_epoch": int(best["epoch"]),
        "best_loss": float(best["loss"]),
        "components": {
            c: float(best[c])
            for c in ("recon", "conservation", "spectral", "gradient",
                      "charge_continuity", "closure_consistency",
                      "energy_conservation", "flow_matching")
            if c in best and best.get(c, "") != ""
        },
        "rmse": dict(zip(CHANNELS, parsePerChannelRmse(
            best.get("per_channel_rmse", "")))),
    }


def fmt(x: float, p: int = 3) -> str:
    return f"{x:.{p}f}"


def fmtLoss(x: float) -> str:
    """Switch to scientific notation for very small losses (the
    hlocalcnn degenerate case prints 0.0000 otherwise)."""
    return f"{x:.4f}" if abs(x) >= 1e-3 else f"{x:.3e}"


def rmseNote(variant: dict) -> str:
    """Per-variant context paragraph appended to the RMSE preamble."""
    if variant["id"] == "hlocalcnn":
        return (" <strong>These values are essentially the noisy-input "
                "passthrough baseline:</strong> the heads stayed near "
                "their zero-init residual because the optimizer drove "
                "the per-scale loss multipliers to ~0 before the "
                "prediction-side gradient signal could shape them.")
    return ""


def convergence_table(variant: dict, stats: dict) -> str:
    comps = stats["components"]
    rows = []
    note = f'Best at epoch {stats["best_epoch"]}.'
    if variant["id"] == "hlocalcnn":
        note += (' The loss collapses to ~10<sup>&minus;7</sup> by '
                 'sending every learnable per-scale / per-cell '
                 'multiplier toward zero; the model itself has not '
                 'learnt anything (per-channel RMSE below matches the '
                 'noisy input).')
    rows.append(
        f'  <tr><td class="label">Total val loss</td>'
        f'<td><strong>{fmtLoss(stats["best_loss"])}</strong></td>'
        f'<td>{note}</td></tr>'
    )
    if variant["id"] == "flowunet":
        flow = comps.get("recon", comps.get("flow_matching", 0.0))
        rows.append(
            f'  <tr><td class="label">Flow-matching MSE</td>'
            f'<td>{fmt(flow, 4)}</td>'
            f'<td>Pointwise MSE between predicted velocity and the '
            f'linear-interpolant velocity (<em>x</em><sub>1</sub> − '
            f'<em>x</em><sub>0</sub>); equals the total loss because '
            f'no other terms are active.</td></tr>'
        )
    else:
        if comps.get("recon", 0) > 0:
            rows.append(
                f'  <tr><td class="label">Recon</td>'
                f'<td>{fmt(comps["recon"], 4)}</td>'
                f'<td>Per-channel-weighted Charbonnier on the '
                f'17 output channels.</td></tr>'
            )
        if comps.get("spectral", 0) > 0:
            rows.append(
                f'  <tr><td class="label">Spectral</td>'
                f'<td>{fmt(comps["spectral"], 4)}</td>'
                f'<td>L<sub>2</sub> in Fourier space.</td></tr>'
            )
        if comps.get("charge_continuity", 0) > 0:
            rows.append(
                f'  <tr><td class="label">Charge continuity</td>'
                f'<td>{fmt(comps["charge_continuity"], 4)}</td>'
                f'<td>Lowpass-filtered (∇·<strong>J</strong><sub>'
                f'pred</sub> − ∇·<strong>J</strong><sub>GT</sub>) at '
                f'cutoff 0.5 Nyquist.</td></tr>'
            )
        if comps.get("closure_consistency", 0) > 0:
            rows.append(
                f'  <tr><td class="label">Closure consistency</td>'
                f'<td>{fmt(comps["closure_consistency"], 4)}</td>'
                f'<td>Per-cell normalized closure-relation '
                f'consistency on (<em>p</em>, <em>P</em>) per species.'
                f'</td></tr>'
            )
        if comps.get("conservation", 0) > 0:
            rows.append(
                f'  <tr><td class="label">Conservation</td>'
                f'<td>{fmt(comps["conservation"], 4)}</td>'
                f'<td>Per-cell mass conservation on (num, p) per '
                f'species.</td></tr>'
            )
        if comps.get("gradient", 0) > 0:
            rows.append(
                f'  <tr><td class="label">Gradient-of-error</td>'
                f'<td>{fmt(comps["gradient"], 4)}</td>'
                f'<td>L<sub>2</sub> on the spatial gradient of the '
                f'residual (pred − GT). Sharpens edges.</td></tr>'
            )
    return ('<table>\n'
            '  <tr><th class="label">Term</th><th>Value at best epoch</th>'
            '<th>Notes</th></tr>\n' + '\n'.join(rows) + '\n</table>')


def rmse_table(stats: dict) -> str:
    rmse = stats["rmse"]
    blocks = [
        ("Currents <em>J</em>",
         [("jx", "Physical mean ≈ 0 by symmetry; RMSE ≈ 1 is the "
                  "constant-mean prediction."),
          ("jy", "Same as jx."),
          ("jz", "In-plane axial current; the only J component with "
                 "coherent structure.")]),
        ("Number density",
         [("num_electrons", "Smooth large-scale density; the residual "
                            "from the constant-mean baseline is what "
                            "the model has to learn."),
          ("num_deuterium", "Same as num_electrons.")]),
        ("Momentum density (electrons)",
         [("px_electrons", "Radial in-plane; carries coherent "
                           "compressive flow."),
          ("py_electrons", "Out-of-plane; physically zero (constant-"
                           "mean baseline)."),
          ("pz_electrons", "Axial in-plane; physically zero "
                           "(constant-mean baseline).")]),
        ("Momentum density (deuterium)",
         [("px_deuterium", "Same role as px_electrons."),
          ("py_deuterium", "Same as py_electrons; tiny GT stddev "
                           "amplifies normalised RMSE."),
          ("pz_deuterium", "Same as pz_electrons.")]),
        ("Kinetic-energy density (electrons)",
         [("enex_electrons", "Cleanly recovered."),
          ("eney_electrons", "Cleanly recovered."),
          ("enez_electrons", "Cleanly recovered.")]),
        ("Kinetic-energy density (deuterium)",
         [("enex_deuterium", "Cleanly recovered."),
          ("eney_deuterium", "Cleanly recovered."),
          ("enez_deuterium", "Cleanly recovered.")]),
    ]
    rows = []
    rows.append('  <tr><th class="label">Channel</th>'
                '<th>RMSE</th><th>Reading</th></tr>')
    for header, items in blocks:
        rows.append(f'  <tr><td class="label" colspan="3" '
                    f'style="background:#f4f4f4">'
                    f'<strong>{header}</strong></td></tr>')
        for ch, note in items:
            v = rmse.get(ch, float("nan"))
            v_str = fmt(v, 3) if v == v else "—"
            rows.append(f'  <tr><td class="label"><code>{ch}</code></td>'
                        f'<td>{v_str}</td><td>{note}</td></tr>')
    return ('<table>\n' + '\n'.join(rows) + '\n</table>')


def comparison_block(variant_id: str) -> str:
    """Per-tier cascading details with one 4x4 figure per channel.

    The 4x4 figure has rows = the 4 canonical phase snapshots and
    columns = (low-ppc input, prediction, ppc1024 ground truth,
    relative error = (pred - GT) / RMS(GT)). The relative error
    column is on a symmetric +/- scale in units of one RMS(GT)."""
    out = []
    all_chs = list(CHANNELS) + list(DERIVED_CHANNELS)
    out.append('<h4>Comparisons by ppc tier</h4>')
    out.append('<p>One figure per (tier, channel). Rows are the four '
               'canonical phases; columns are low-ppc input, model '
               'prediction, ppc<sub>1024</sub> ground truth, and '
               'relative error <em>(pred &minus; GT) / RMS(GT)</em>. '
               'Pressure (<em>P<sub>xx</sub></em>, <em>P<sub>yy</sub>'
               '</em>, <em>P<sub>zz</sub></em>) and temperature '
               '(<em>T<sub>x</sub></em>, <em>T<sub>y</sub></em>, '
               '<em>T<sub>z</sub></em>) are derived from the predicted '
               'moments via the closure formula and compared against '
               'the same closure on the GT moments.</p>')
    for tidx, tier in enumerate(TIERS):
        opn = ' open' if tidx == 0 else ''
        out.append(f'<details class="comparison-block"{opn}>')
        out.append(f'<summary>{TIER_LABEL[tier]}</summary>')
        for ch in all_chs:
            src = (f'{FIG_DIR}/comparison_{variant_id}_{tier}_{ch}.png')
            alt = f'{variant_id}: {ch}, {tier}'
            out.append('<figure>')
            out.append(f'  <img loading="lazy" decoding="async" '
                       f'src="{src}" alt="{alt}">')
            out.append(f'  <figcaption><code>{ch}</code> at {tier} '
                       f'across the four canonical phases.'
                       f'</figcaption>')
            out.append('</figure>')
        out.append('</details>')
    return '\n'.join(out)


def variant_section(variant: dict) -> str:
    metrics = (RUNS_DIR / f'.run_pdn_{variant["run_dir"]}.matrix' /
               'metrics.csv')
    stats = extract(metrics)
    vid = variant["id"]

    # Variants whose v1 result is documented but whose figures are
    # not regenerated in the current 4x4-grid / 2x2-RMSE format
    # (because the v1 checkpoint is degenerate and the v2 architecture
    # cannot meaningfully load it). Keep the convergence + RMSE
    # tables but drop the chart figures and the comparison block.
    figures_only_in_v1 = {"hlocalcnn"}

    if vid in figures_only_in_v1:
        return f"""

<h2 id="{variant['anchor']}">{variant['section']} {variant['title']}</h2>
<p>{variant['loss_intro']}</p>

<h4>Convergence</h4>
{convergence_table(variant, stats)}

<h4>Per-channel RMSE at best epoch</h4>
<p>All values in channel-stddev units; RMSE = 1 corresponds to the constant-mean prediction. RMSE &lt; 1 means the model beats the constant-mean baseline; RMSE &gt; 1 means it is worse.{rmseNote(variant)}</p>
{rmse_table(stats)}

<p><em>Comparison plots and the RMSE bar chart are omitted for this
variant: the v1 checkpoint is degenerate (loss collapsed to ~10
<sup>&minus;7</sup> with no useful learning), so per-cell comparison
panels would just show the noisy input. A v2 run on the
softmax-with-fixed-total parameterisation is in flight.</em></p>
"""

    return f"""

<h2 id="{variant['anchor']}">{variant['section']} {variant['title']}</h2>
<p>{variant['loss_intro']}</p>

<h4>Convergence</h4>
{convergence_table(variant, stats)}

<figure>
  <img loading="lazy" decoding="async" src="{FIG_DIR}/loss_{vid}.png" alt="Loss curves: {variant['title']}">
  <figcaption>{variant['title']}: per-epoch loss curves (train, val) for the total and each enabled term.</figcaption>
</figure>

<h4>Per-channel RMSE at best epoch</h4>
<p>All values in channel-stddev units; RMSE = 1 corresponds to the constant-mean prediction. RMSE &lt; 1 means the model beats the constant-mean baseline; RMSE &gt; 1 means it is worse.{rmseNote(variant)}</p>
{rmse_table(stats)}

<figure>
  <img loading="lazy" decoding="async" src="{FIG_DIR}/rmse_{vid}.png" alt="Per-channel RMSE: {variant['title']}">
  <figcaption>{variant['title']}: per-channel RMSE start-vs-best, split into signal-carrying channels (jz, num, pz, P, T) and noisy / baseline channels (jx, jy, px, py, ene).</figcaption>
</figure>

{comparison_block(vid)}
"""


def main() -> None:
    parts = []
    parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>4.3.2 Denoisers on the current channel set</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<div class="layout">

<aside class="sidebar" data-page="status_denoisers"></aside>

<main class="content">

<h1>4.3.2 Denoisers on the current channel set</h1>

<p class="intro">
Training results for three denoiser variants on the SNR-restricted
channel set with per-species momentum-density and kinetic-energy
outputs. Design choices documented in
<a href="approach.html#asymmetric">§3.7</a>,
<a href="approach.html#snr-channel-selection">§3.8</a>, and
<a href="approach.html#loss-formulation">§3.10</a>; this page reports
the outcomes. Three variants trained on Matrix (1×MI300 GPU, bf16
AMP):
<a href="#charbonnier">§4.3.2.1 Global U-Net / Charbonnier</a>,
<a href="#charbonnier-grad">§4.3.2.2 Global U-Net / Charbonnier + gradient-of-error</a>,
<a href="#localcnn">§4.3.2.3 Local CNN / Charbonnier</a>,
<a href="#hlocalcnn">§4.3.2.4 Hierarchical Local CNN / Charbonnier (degenerate)</a>,
and <a href="#flowunet">§4.3.2.5 Flow-matching U-Net (linear interpolant)</a>;
<a href="#rationale">§4.3.2.6 Design rationale</a> summarises why
these are the levers being pulled.
</p>
""")
    for v in VARIANTS:
        parts.append(variant_section(v))
    parts.append("""

<h2 id="rationale">4.3.2.6 Design rationale</h2>

<p>The five variants probe orthogonal levers on the same supervised
task:</p>

<ul>
  <li><strong>Charbonnier baseline.</strong> Robust pixel loss with the
  cross-channel terms active. Sets the regression-style ceiling for
  the global U-Net.</li>
  <li><strong>+ gradient-of-error.</strong> Adds an L<sub>2</sub> on
  the spatial gradient of the residual, sharpening edges. Diagnoses
  whether smoothing in the baseline is dominated by the loss or by
  the network.</li>
  <li><strong>Local CNN.</strong> 7&times;7 input patch &rarr;
  single centre-cell prediction with a small CNN. The local-physics
  inductive bias (PIC noise is per-cell statistical noise) makes
  this a natural fit and the model trains faster than the global
  U-Net at the same per-snapshot fidelity. See
  <a href="approach_localParticleDenoiser.html">&sect;3.2</a>.</li>
  <li><strong>Hierarchical-readout local CNN.</strong> Same patch
  contract as the Local CNN, plus intermediate-scale readouts that
  let spatial-derivative loss terms re-enter the local path. The
  v1 implementation collapsed to a degenerate solution
  (see &sect;4.3.2.4); the learnable per-scale multipliers need
  a constraint that prevents them from being driven to zero.</li>
  <li><strong>Flow-matching U-Net.</strong> Replaces the
  regression-style mapping with an ODE-integrated generative
  trajectory from <em>x</em><sub>0</sub> (noisy) to <em>x</em>
  <sub>1</sub> (clean). Tests whether learning a velocity field is
  better-conditioned than learning the residual directly. See
  <a href="approach_flowParticleDenoiser.html">&sect;3.6</a>.</li>
</ul>

<p>The local KPCN variant is still in flight; its results will appear
under §4.3.3 once that run lands.</p>

</main>

</div>
<script src="nav.js"></script>
</body>
</html>
""")
    out_path = DOCS_DIR / "status_denoisers.html"
    out_path.write_text("".join(parts))
    print(f"wrote {out_path}  ({sum(len(p) for p in parts)} chars)")


if __name__ == "__main__":
    main()
