const tf = require("@tensorflow/tfjs-node");
// const tf = require('@tensorflow/tfjs-node-gpu');
const _ = require('lodash');
const cv = require('opencv4nodejs');
 
const pascalvoc = [[ 192,192,192 ],[ 105,105,105 ],
                    [ 160,82,45 ],[ 244,164,96 ],[ 60,179,113 ],
                    [ 34,139,34 ],[ 154,205,50 ],[ 0,128,0 ],
                    [ 0,100,0 ],[ 0,250,154 ],[ 139,69,19 ],
                    [ 1,51,73 ],[ 190,153,153 ],[ 0,132,111 ],
                    [ 0,0,142 ],[ 0,60,100 ],[ 135,206,250 ],
                    [ 128,0,128 ],[ 153,153,153 ],[ 255,255,0 ],
                    [ 220,20,60 ],[ 255,182,193 ],[ 220,220,220 ],[ 0,0,0 ]];

const modelDir = 'model';
const interval = 1000;

async function load_model() {
    return await tf.loadLayersModel(`file://${modelDir}/model.json`);
}

const modelPromise = load_model();

const cap = new cv.VideoCapture('video.avi');

setInterval(() => {
    const frame = cap.read();
    // const frame = cv.imread(files[15])
    const resize = frame.resize(128, 128);

    Promise.all([modelPromise])
    .then(values => {
        detectFrame(resize, values[0]);
    })
    .catch(error => {
        console.error(error);
    });

}, interval);

detectFrame = (video, model) => {
    tf.engine().startScope();
    const predictions = model.predict(process_input(video));
    const f = renderPredictions(predictions);
    Promise.resolve(f).then(function(mat){
        const resize = mat.resize(420, 680);
        cv.imshow('Segmentation', resize);
        cv.waitKey(interval);
    });
    tf.engine().endScope();
};

function process_input(video_frame) {
    const width = 128;
    const height = 128;
    const img = tf.tensor3d(new Uint8Array(video_frame.getData()), [width, height, 3], 'float32');
    const scale = tf.scalar(255.);
    const normalised = img.div(scale);
    const batched = normalised.expandDims();
    return batched;
};

renderPredictions = async (predictions) => {
    const img_shape = [128, 128]
    const offset = 0;
    const segmPred = tf.image.resizeBilinear(predictions, img_shape);
    const segmMask = segmPred.argMax(3).reshape(img_shape);
    const width = segmMask.shape.slice(0, 1);
    const height = segmMask.shape.slice(1, 2);
    const data = await segmMask.data();
    const bytes = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < height * width; ++i) {
      const partId = data[i];
      const j = i * 4;
      if (partId === -1) {
          bytes[j + 0] = 255;
          bytes[j + 1] = 255;
          bytes[j + 2] = 255;
          bytes[j + 3] = 255;
      } else {
          const color = pascalvoc[partId + offset];

          if (!color) {
              throw new Error(`No color could be found for part id ${partId}`);
          }
          bytes[j + 0] = color[0];
          bytes[j + 1] = color[1];
          bytes[j + 2] = color[2];
          bytes[j + 3] = 255;
      }
    }

    const normalArray = Array.from(bytes);
    const channels = 4;
    const nestedChannelArray = _.chunk(normalArray, channels);
    const nestedImageArray = _.chunk(nestedChannelArray, height);

    const RGBAmat = new cv.Mat(nestedImageArray, cv.CV_8UC4);

    const BGRAmat = RGBAmat.cvtColor(cv.COLOR_RGBA2BGRA);

    return await BGRAmat;
};