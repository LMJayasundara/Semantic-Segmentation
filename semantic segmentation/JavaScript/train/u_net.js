const tf = require("@tensorflow/tfjs-node");

const nfilters = 64;

function conv_block (tensor, nfilters, size=3){
    var x = tf.layers.conv2d({filters: nfilters, kernelSize: [size, size], padding: "same", kernelInitializer: "heNormal"}).apply(tensor);
    var x = tf.layers.batchNormalization().apply(x);
    var x = tf.layers.activation({activation: 'relu'}).apply(x);
    var x = tf.layers.conv2d({filters: nfilters, kernelSize: [size, size], padding: "same", kernelInitializer: "heNormal"}).apply(x);
    var x = tf.layers.batchNormalization().apply(x);
    var x = tf.layers.activation({activation: 'relu'}).apply(x);
    return x;
}

function deconv_block (tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)){
    var y = tf.layers.conv2dTranspose({filters: nfilters, kernelSize: [size, size], padding: "same", strides: [2, 2]}).apply(tensor);
    var y = tf.layers.concatenate().apply([y, residual]);
    var y = conv_block(y, nfilters);
    return y;

}

const input_layer = tf.input({ shape: [128, 128, 3], name: 'image_input'});
const conv1 = conv_block(input_layer, nfilters);
const conv1_out = tf.layers.maxPooling2d({poolSize: [2, 2]}).apply(conv1);
const conv2 = conv_block(conv1_out, nfilters*2);
const conv2_out = tf.layers.maxPooling2d({poolSize: [2, 2]}).apply(conv2);
const conv3 = conv_block(conv2_out, nfilters*4);
const conv3_out = tf.layers.maxPooling2d({poolSize: [2, 2]}).apply(conv3);
const conv4 = conv_block(conv3_out, nfilters*8);
const conv4_out = tf.layers.maxPooling2d({poolSize: [2, 2]}).apply(conv4);
const conv4_dout = tf.layers.dropout({ rate: 0.5 }).apply(conv4_out);
const conv5 = conv_block(conv4_dout, nfilters*16);
const dconv5 = tf.layers.dropout({ rate: 0.5 }).apply(conv5);

const deconv6 = deconv_block(dconv5, residual=conv4, nfilters*8);
const ddeconv6 = tf.layers.dropout({ rate: 0.5 }).apply(deconv6);
const deconv7 = deconv_block(ddeconv6, residual=conv3, nfilters*4);
const ddeconv7 = tf.layers.dropout({ rate: 0.5 }).apply(deconv7);
const deconv8 = deconv_block(ddeconv7, residual=conv2, nfilters*2);
const deconv9 = deconv_block(deconv8, residual=conv1, nfilters);
const output_layer = tf.layers.conv2d({filters: 32, kernelSize: [1, 1], activation: 'softmax'}).apply(deconv9);

const model = tf.model({ inputs: input_layer, outputs: output_layer });

model.compile({
  loss: tf.losses.softmaxCrossEntropy,
  optimizer: 'adam',
  metrics: 'accuracy'
});

model.summary();