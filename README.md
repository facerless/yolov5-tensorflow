# yolov5-tensorflow


1.生成参数文件  
gen_tf_params.py  
将该py文件拷贝至yolov5官方源码主目录下，再运行该py。  


2.转换成tf pb模型  
yolov5_tf.py  
目前支持tf1.13，tf1.15  
转换时需要确保class_num、anchors与转换时使用的的参数一致  
本代码目前支持v2.0和v3.0版本代码，需要自行设置convBnLeakly函数中注释行代码  

