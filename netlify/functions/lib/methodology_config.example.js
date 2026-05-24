// Committed skeleton — no methodology content.
// Real file is gitignored. Inject via METHODOLOGY_CONFIG env var on Netlify (base64 JSON).
// See CLAUDE.md "Privacy" section for the model.

module.exports = {
  archetypes: {
    TYPE_A: { name: '', short: '', desc: '', action: '', signal_keys: [] },
    TYPE_B: { name: '', short: '', desc: '', action: '', signal_keys: [] },
    TYPE_C: { name: '', short: '', desc: '', action: '', signal_keys: [] },
    TYPE_D: { name: '', short: '', desc: '', action: '', signal_keys: [] },
  },
  rthBias: {
    BULLISH: { label: '', cls: 'bull',  summary: '' },
    BEARISH: { label: '', cls: 'bear',  summary: '' },
    NEUTRAL: { label: '', cls: 'mixed', summary: '' },
    UNKNOWN: { label: '', cls: 'ghost', summary: '' },
  },
  yieldSignals: {
    RISING_FAST: { label: '', cls: 'bear',  interp: '' },
    RISING:      { label: '', cls: 'bear',  interp: '' },
    STABLE:      { label: '', cls: 'mixed', interp: '' },
    FALLING:     { label: '', cls: 'bull',  interp: '' },
    UNAVAILABLE: { label: '', cls: 'ghost', interp: '' },
  },
  bojSignals: {
    CARRY_UNWIND:  { label: '', cls: 'bear',  interp: '' },
    YEN_STABLE:    { label: '', cls: 'mixed', interp: '' },
    YEN_WEAKENING: { label: '', cls: 'bull',  interp: '' },
    UNAVAILABLE:   { label: '', cls: 'ghost', interp: '' },
  },
  cotLabels: {
    FUMES_LONG:    { label: '', cls: 'bear',  interp: '' },
    EXTREME_SHORT: { label: '', cls: 'bull',  interp: '' },
    NEUTRAL:       { label: '', cls: 'mixed', interp: '' },
    UNAVAILABLE:   { label: '', cls: 'ghost', interp: '' },
  },
  liquidityLabels: {
    IMPROVING:     { label: '', cls: 'bull',  interp: '' },
    STABLE:        { label: '', cls: 'mixed', interp: '' },
    DETERIORATING: { label: '', cls: 'bear',  interp: '' },
    UNAVAILABLE:   { label: '', cls: 'ghost', interp: '' },
  },
};
