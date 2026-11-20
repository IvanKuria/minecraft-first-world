# minecraft-first-world

Code from my attempt to recover the world shown in Notch's first Minecraft video, the "Cave game tech test" uploaded on May 13, 2009 (YouTube id `F9t3FREAZ-k`, about 181 seconds).

**Status: unsolved.** No seed, no complete world and no original cave generator has been recovered. What's here is the tooling and the partial results, shared so other people can push it further.

## What the footage gives you

- The end of the video briefly shows Notch's editor. That frame shows `gluPerspective(70, ...)` and `new Level(256, 256, 64)`, so the field of view is 70° and the world is 256 × 256 × 64.
- The player presses R (reset/respawn) twice on camera. Each reset places the player at a random position drawn from Java's `java.util.Random`, so precise enough measurements of those positions can, in principle, pin down the RNG state.
- The later part of the video falls outside the world and shows its edges and underside, which gives an absolute reference for coordinates.

## What's in here

| folder | what it does |
|---|---|
| `runtime/` | Runs a preserved early build with probes (`EarlyWorldProbe`, `GameRenderProbe`, `ValidateCandidateStates`). Includes the RNG work: lattice recovery of Java's 48-bit `Random` state from noisy coordinates, the mixed-modulus solver (rough absolute + precise fractional coordinates), and enumeration/scoring of candidate states. |
| `geometry/` | Camera calibration and pose fitting from the footage: renderer-accurate texture projection, boundary fits against the 256-block world edge, respawn-position estimates with uncertainty. |
| `timing/` | Frame timing, respawn comparison and motion analysis around the resets. |
| `generator/` | Experiments on the missing generator: growth inference, opening and first-reset texture matching, chunk build order (with small LWJGL input stubs so original classes run headless). |
| `proof/` | Later search experiments: CPU and GPU (CUDA) seed searches over candidate generators (`c5`, `pc-c5`), cave/terrain feature fitting (`caves`, `terrace`), likelihood scoring and planted-seed validation (searches are tested on synthetic worlds with a known seed before being run on the footage). |

## Results so far

- Correct pixel-centre sampling reproduces the preserved renderer at correlation 0.998, and the calibrated projection gets strong grass-texture matches after both resets, including held-out patches.
- Synthetic tests recover the full 48-bit `Random` state from three noisy spawns (±0.1 block) in 64/64 trials. Two spawns at that precision usually leave false candidates.
- Using only the two filmed resets with the mixed-modulus solver gives a conditional family of 63,074 internal states. An independent fit of the opening view narrows that to 175, but about 173 matches are expected by chance, so this does **not** identify the seed.
- The biggest blockers: the original cave generator was never released, the first reset's position before movement is uncertain, and the RNG calls between world generation and the resets are unknown.

## What you need that isn't here

This repo contains only my own code. It does not include the video, game files, decompiled sources or other people's projects. You'll need:
- a copy of the footage (the original upload and several re-uploads are archived),
- an early preserved Minecraft build (rd-era versions are archived by the community),
- Python 3 with numpy/scipy/opencv, a JDK, and for the GPU search an NVIDIA GPU with CUDA.

Many scripts were written as one-off experiments and expect their inputs next to them; read the top of each script before running it.

## If you get further

If you narrow it down, I'd love to hear about it. Open an issue or reach out through my YouTube channel.

License: MIT.
