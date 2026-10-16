// ONNX Runtime Web adapter for the exported Halloween dodge student.

export async function loadDodgePolicy({baseUrl = '../model'}) {
  if (!globalThis.ort) {
    throw new Error('onnxruntime-web is not loaded');
  }
  const meta = await fetch(`${baseUrl}/halloween_dodge.json`).then((r) => {
    if (!r.ok) throw new Error(`metadata fetch failed: ${r.status}`);
    return r.json();
  });
  const session = await ort.InferenceSession.create(`${baseUrl}/${meta.onnx_file}`, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  return new DodgePolicy(session, meta);
}

export class DodgePolicy {
  constructor(session, meta) {
    this.session = session;
    this.meta = meta;
    this.mean = Float32Array.from(meta.obs_mean);
    this.std = Float32Array.from(meta.obs_std);
    this.input = new Float32Array(meta.input_dim);
  }

  async act(state, command) {
    const {state_dim, command_dim, input_dim} = this.meta;
    if (state.length !== state_dim) throw new Error(`state ${state.length} != ${state_dim}`);
    if (command.length !== command_dim) throw new Error(`command ${command.length} != ${command_dim}`);
    for (let i = 0; i < state_dim; i++) this.input[i] = state[i];
    for (let i = 0; i < command_dim; i++) this.input[state_dim + i] = command[i];
    for (let i = 0; i < input_dim; i++) this.input[i] = (this.input[i] - this.mean[i]) / this.std[i];
    const feeds = {obs: new ort.Tensor('float32', this.input, [1, input_dim])};
    const out = await this.session.run(feeds);
    return out.action.data;
  }
}
