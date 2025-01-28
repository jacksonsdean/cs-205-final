import * as tf from '@tensorflow/tfjs';
async function main() {
    function randomWeight(std = 1.0) {
      return tf.variable(tf.scalar((Math.random()*2 - 1)*std, 'float32'));
    }
    
    class SimpleCPPN {
      constructor(numInputs, numHidden, numOutputs) {
        // Create arrays to hold weights, biases, etc.
        this.W_ih = tf.variable(tf.randomNormal([numInputs, numHidden], 0, 1.0));
        this.b_ih = tf.variable(tf.zeros([numHidden]));
        this.W_ho = tf.variable(tf.randomNormal([numHidden, numOutputs], 0, 1.0));
        this.b_ho = tf.variable(tf.zeros([numOutputs]));
      }
    
      forward(x) {
        // x shape: [batch, numInputs]
        let hidden = x.matMul(this.W_ih).add(this.b_ih); // shape: [batch, numHidden]
        hidden = tf.sin(hidden); // pick an activation
        let out = hidden.matMul(this.W_ho).add(this.b_ho); // shape: [batch, numOutputs]
        out = tf.tanh(out);
        return out; // shape: [batch, numOutputs]
      }
    
      mutate(rate, std) {
        // For each param, with some probability, add random noise:
        this._mutateVar(this.W_ih, rate, std);
        this._mutateVar(this.b_ih, rate, std);
        this._mutateVar(this.W_ho, rate, std);
        this._mutateVar(this.b_ho, rate, std);
      }
    
      _mutateVar(variable, rate, std) {
        tf.tidy(() => {
          const shape = variable.shape;
          const r = tf.randomUniform(shape); // uniform [0,1]
          const mask = tf.where(r.less(rate), tf.onesLike(r), tf.zerosLike(r));
          const noise = tf.randomNormal(shape, 0, std);
          const update = variable.add(mask.mul(noise));
          variable.assign(update);
        });
      }
    }
    
    // usage example:
    const net = new SimpleCPPN(3, 10, 1);
    const x = tf.randomNormal([5,3]); // 5 examples, 3 inputs each
    const y = net.forward(x);
    
    y.print();
    net.mutate(0.1, 0.5);
}
  main();