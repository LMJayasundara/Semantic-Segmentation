const tf = require("@tensorflow/tfjs-node");
const cv = require('opencv4nodejs');

const frame = cv.imread('test.png');
const imgBase64 =  cv.imencode('.png', frame).toString('base64');
const trainImageBuffer = Buffer.from(imgBase64, 'base64');
const mask = tf.node.decodePng(trainImageBuffer);

const colors = [[ 192,192,192 ],[ 105,105,105 ],
                [ 160,82,45 ],[ 244,164,96 ],[ 60,179,113 ],
                [ 34,139,34 ],[ 154,205,50 ],[ 0,128,0 ],
                [ 0,100,0 ],[ 0,250,154 ],[ 139,69,19 ],
                [ 1,51,73 ],[ 190,153,153 ],[ 0,132,111 ],
                [ 0,0,142 ],[ 0,60,100 ],[ 135,206,250 ],
                [ 128,0,128 ],[ 153,153,153 ],[ 255,255,0 ],
                [ 220,20,60 ],[ 255,182,193 ],[ 220,220,220 ],[ 0,0,0 ]];

var one_hot_map = [];

for (const color in colors) {
    class_map = tf.all(tf.equal(mask, colors[color]), axis=-1);
    one_hot_map.push(class_map);
}

var one_hot_map = tf.stack(one_hot_map, axis=-1);
var one_hot_map = tf.cast(one_hot_map, 'float32');

var maskImg = tf.argMax(one_hot_map, axis=-1, 'int32');

const one_hot = tf.oneHot(maskImg, 24).print();