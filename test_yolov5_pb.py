import tensorflow as tf
import numpy as np
import cv2 as cv
import time

def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    '''
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    # print(img.dtype,img.shape)
    # print(c1,c2)
    cv.rectangle(img, c1, c2, color)#, thickness=2
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]

        x1,y1=c1[0],c1[1]
        y1=y1 if y1>20 else y1+20
        x2,y2=x1+t_size[0],y1-t_size[1]

        cv.rectangle(img, (x1,y1), (x2,y2), color, -1)  # filled

        cv.putText(img, label, (x1, y1), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv.LINE_AA)

if __name__=="__main__":
    pb_path=r'yolov5.pb'
    img_path=r'1610935620(1).jpg'
    input_shape=[800,800]


    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        # with tf.Session() as sess:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        input = sess.graph.get_tensor_by_name("input:0")
        output = sess.graph.get_tensor_by_name("output:0")

        img = cv.imdecode(np.fromfile(img_path, np.uint8), -1)[:, :, :3]
        img = cv.resize(img, tuple(input_shape[::-1]))
        img_ = np.expand_dims(img[:, :, ::-1], 0).astype(np.float32) / 255.

        for i in range(50):
            t1=time.clock()
            pred = sess.run(output, feed_dict={input: img_})
            print(time.clock()-t1)
        np.set_printoptions(suppress=True)
        print(pred)
        for i in range(pred.shape[0]):
            box = pred[i, :4].astype(np.int32)
            score = pred[i, 4]
            label = int(pred[i, 5])
            plot_one_box(img, box, '%d_%.4f' % (label, score), (0, 255, 0))
        cv.imshow("result", img)
        cv.waitKey(0)


        print(tf.__version__)