# Ablation Notes

Important event ablations from the 0411 codebase:

- RGB-only baseline: no event tensor, no event-edge guidance.
- EventFusion: 10-channel old/new voxel tensor plus 4 event-stat channels.
- Event reliability: density, temporal balance, and polarity balance gate event
  influence.
- Event-edge adapter: 3 channels per event radius, commonly `[25, 50, 100]` or
  `[25, 50, 200]`.
- 50ms-only ablation: `EDGE_WINDOW_RADII_MS: [50]`, 3 channels.
- Zero-event control: keeps the event code path but zeros event input.
