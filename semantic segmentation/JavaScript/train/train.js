const glob = require('glob');
const tf = require("@tensorflow/tfjs-node");
const fs = require('fs');

const cv = require('opencv4nodejs');

const train = glob.sync('data/train/*.png');
const train_labels_ids = glob.sync('data/train_labels/*.png');

const data = train.map(function(c, i) { return [ c, train_labels_ids[i] ] });

const modelDir = 'model';

const colors = [[ 192,192,192 ],[ 105,105,105 ],
                [ 160,82,45 ],[ 244,164,96 ],[ 60,179,113 ],
                [ 34,139,34 ],[ 154,205,50 ],[ 0,128,0 ],
                [ 0,100,0 ],[ 0,250,154 ],[ 139,69,19 ],
                [ 1,51,73 ],[ 190,153,153 ],[ 0,132,111 ],
                [ 0,0,142 ],[ 0,60,100 ],[ 135,206,250 ],
                [ 128,0,128 ],[ 153,153,153 ],[ 255,255,0 ],
                [ 220,20,60 ],[ 255,182,193 ],[ 220,220,220 ],[ 0,0,0 ]];

async function* dataGenerator() {
    while (true) {
        
        for await (const path of data) {
            const frame = cv.imread(path[0]);
            const framex = frame.resize(128, 128);
            const imgBase64 =  cv.imencode('.png', framex).toString('base64');
            const trainImageBuffer = Buffer.from(imgBase64, 'base64');
            // cv.imshowWait('', frame)

            const lblframe = cv.imread(path[1]);
            const lblframex = lblframe.resize(128, 128);
            const lblBase64 =  cv.imencode('.png', lblframex).toString('base64');
            const trainLblImageBuffer = Buffer.from(lblBase64, 'base64');
            // cv.imshowWait('', lblframe)

            var mask = tf.node.decodePng(trainLblImageBuffer);
            var one_hot_map = [];
            for (var color in colors) {
                class_map = tf.all(tf.equal(mask, colors[color]), axis=-1);
                one_hot_map.push(class_map);
            }
            var one_hot_mapy = tf.stack(one_hot_map, axis=-1);
            var one_hot_mapx = tf.cast(one_hot_mapy, 'float32');
            var maskx = tf.argMax(one_hot_mapx, axis=-1, 'int32');
            var one = tf.oneHot(maskx, 24);

            // const trainImageBuffer = fs.promises.readFile(path[0]);
            // const trainLblImageBuffer = fs.promises.readFile(path[1]);

            yield [await trainImageBuffer, await one];
        }
        
    }
}

async function initModel() {
    let model;
  
    try {
        model = await tf.loadLayersModel(`file://${modelDir}/model.json`);
        console.log(`Model loaded from: ${modelDir}`);
    } catch {
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
        const output_layer = tf.layers.conv2d({filters: 24, kernelSize: [1, 1], activation: 'softmax'}).apply(deconv9);
        
        model = tf.model({ inputs: input_layer, outputs: output_layer });
        
    }
  
    model.compile({
        loss: tf.losses.softmaxCrossEntropy,
        optimizer: 'adam',
        metrics: 'accuracy'
    });
  
    return model;
}

(async function () {
    
    const batchSize = 4;
    const epochs = 2;
    const totalSamples = 400;


    const dataset = tf.data
    .generator(dataGenerator)
    .map(([imageBuffer, lblBuffer]) => {
      
        // const image = cv.imdecode(imageBuffer)
        // cv.imshowWait('', image)
        // const lbl = cv.imdecode(lblBuffer)
        // cv.imshowWait('', lbl)

        const xs = tf.node.decodePng(imageBuffer).div(255);
        const ys = lblBuffer;

        return {xs, ys};
    })
    .shuffle(batchSize)
    .batch(batchSize);

    const model = await initModel();
    model.summary();

    // await dataset.forEachAsync(function (dataset) {
    //     console.log(dataset);
    // });

    await model.fitDataset(dataset, {
        epochs,
        batchesPerEpoch: Math.floor(totalSamples / batchSize)
    });

    await model.save(`file://${modelDir}`);

    console.log(`Model saved to: ${modelDir}`);

})();