import tensorflow as tf
import cv2

# imagePayload = tf.io.read_file("./data/test_labels/1584364568817773824.png")
imagePayload = cv2.imread("test.png")
img_str = cv2.imencode('.png', imagePayload)[1].tostring()
mask = tf.io.decode_png(img_str)

colors = [( 192,192,192 ),( 105,105,105 ),
          ( 160,82,45 ),( 244,164,96 ),( 60,179,113 ),
          ( 34,139,34 ),( 154,205,50 ),( 0,128,0 ),
          ( 0,100,0 ),( 0,250,154 ),( 139,69,19 ),
          ( 1,51,73 ),( 190,153,153 ),( 0,132,111 ),
          ( 0,0,142 ),( 0,60,100 ),( 135,206,250 ),
          ( 128,0,128 ),( 153,153,153 ),( 255,255,0 ),
          ( 220,20,60 ),( 255,182,193 ),( 220,220,220 ),( 0,0,0 )]


one_hot_map = []
for color in colors:
    class_map = tf.reduce_all(tf.equal(mask, color), axis=-1)
    one_hot_map.append(class_map)

one_hot_map = tf.stack(one_hot_map, axis=-1)
one_hot_map = tf.cast(one_hot_map, tf.float32)

mask = tf.argmax(one_hot_map, axis=-1)

tf.print(mask)