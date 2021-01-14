
Started with the project
========================

### First steps 

If you start with the project you should follow:

* Installation Option 1
* Use of Pre-trained models (YoloV3 coco weights)
* Inferences by using the webcam if you have one or image folder.


Then you can start using the pipeline. 

### Pipeline description 

Images+Annotations folders --> generate .tfreccord files --> generate .tf checkpoint --> make inferences

Example : Download Images+Annotations folders with OpenImageV4, generate tfreccord, .tf file and make inferences:

main.py downloader --classes Apple Orange --type_csv all --limit 1000 --> generate-tfrecord.py --> train_tfrecord_dataset.py --> detect_video.py --video 0 --classes ./data/oidv4.names


where oidv4.names:

Apple
Orange

To start with the pipeline you should follow the part:

* Create Dataset with Open Images dataset
* Train your model 
* Inferences



Installation
============

All the codes have been updated for Tensorflow 2.X.

#### Option 1:

conda env create -f env_tf2_gpu_yolov3.yml

#### Option 2:

    conda create --name env_tf_gpu python=3.X (7 at the moment)
    conda install pip
    pip install -r requirements.txt

#### Option 3:

Except special needs, I strongly recommend to use conda to install tensorflow: 

    conda create --name env_tf_gpu python=3.X (7 at the moment)
    conda activate env_tf_gpu
    conda install tensorflow-gpu (Install Cuda, TensorRT...)

You may have the need to install other libraries. Use conda install or pip install in your environnment.

##### Optional for option 3

Already installed in option 1 and 2.

Preview build (unstable version). You can use it in a second time, the prediction time has been reduced from 40 to 28ms with it:

    pip install tf-nightly-gpu

Then you could have a multiple versions problem. run python fix_tensorboard.py in your environment and follow the instructions to fix it.


Use of Pre-trained models
========================

You can download weights directly and converting the weights in .tf to make prediction. This part is also usefull if you want to do transfer learning with pre trained model.
 Here is 3 examples of weights you can use.

#### YoloV3 coco weights (80 classes)

    wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
    python convert.py

#### YoloV3-tiny coco weights

    wget https://pjreddie.com/media/files/yolov3-tiny.weights -O data/yolov3-tiny.weights
    python convert.py --weights ./data/yolov3-tiny.weights --output ./checkpoints/yolov3-tiny.tf --tiny

#### YoloV3 OpenImageV4 weigths (601 classes)

    wget https://pjreddie.com/media/files/yolov3-openimages.weights
    python convert.py --weights ./data/yolov3-openimages.weights --output ./checkpoints/yolov3-openimages.tf --num_classes 601

When the .tf file is download, you can go to the inferences part to try your pre-trained model.
Here is the link to download coco.names : https://github.com/pjreddie/darknet/blob/master/data/coco.names

Create Dataset
==============

The goal of this part is to create .tfreccord files in order to train your model.

Your can use Open Image Dataset to download the object you want to detect (1). If the object is not in the dataset, you have to create your own dataset with images and annotations (2). Finally, you can also train the 20 classes of VOC pascal data set (3).

#### 1: with Open Images dataset

+600 classes are available. You can view the set of boxable classes as a hierarchy here: https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy_visualizer/circle.html
put and underscore _ instead an space for classes with a space.

For example: 

    python main.py downloader --classes Apple Orange --type_csv all --limit 1000
    python main.py downloader --classes Bee --type_csv all

You can also visualize the data with:

    python3 main.py visualizer

After downloading the annotations, you can generate the tfrecords files.

    python generate-tfrecord.py \
    	--classes_file=./OIDv4_ToolKit/OID/oidv4.names \
    	--class_descriptions_file=./OIDv4_ToolKit/OID/csv_folder/class-descriptions-boxable.csv \
    	--annotations_file=./OIDv4_ToolKit/OID/csv_folder/train-annotations-bbox.csv \
    	--images_dir=./OIDv4_ToolKit/OID/Dataset/train \
    	--output_file=./train.tfrecord
    python generate-tfrecord.py \
    	--classes_file=./OIDv4_ToolKit/OID/oidv4.names \
    	--class_descriptions_file=./OIDv4_ToolKit/OID/csv_folder/class-descriptions-boxable.csv \
    	--annotations_file=./OIDv4_ToolKit/OID/csv_folder/validation-annotations-bbox.csv \
    	--images_dir=./OIDv4_ToolKit/OID/Dataset/validation \
    	--output_file=./validation.tfrecord

#### 2: with your custom data

* You can use VoTT to create a database with XML annotation (two folders 'Annotations' and 'JPEGImages')
* Split your dataset between 2 our 3 folders (Train, Validation, test)
* Create a label_map.pbtxt with the classes and ID.

Format example of a label_map.pbtxt :

    	item {
    	  id: 1
    	  name: 'bee'
    	}
    	item {
    	  id: 2
    	  name: 'snake'
    	}

Launch create_tfrecords_from_xml.py

Example:

    python create_tfrecords_from_xml.py   \
    	--image_dir=Hilti_custom_data/six-sided-dice-data/JPEGImages \
    	--annotations_dir=Hilti_custom_data/six-sided-dice-data/Annotations \
    	--label_map_path=Hilti_custom_data/six-sided-dice-data/six-sided-dice_label_map.pbtxt \
    	--output_path=Hilti_custom_data/six-sided-dice-data/data_dice.tfrecord

#### 3: with VOC pascal dataset

    python create_tfrecords_voc2012.py \
    	--data_dir './data/voc2012_raw/VOCdevkit/VOC2012' \
    	--split train \
    	--output_file ./data/voc2012_train.tfrecord
    python create_tfrecords_voc2012.py \
    	--data_dir './data/voc2012_raw/VOCdevkit/VOC2012' \
    	--split val \
    	--output_file ./data/voc2012_val.tfrecord


Train your model
================

In order to train your model by feature learning you have to use a pre trained model. See the part Use of Pre-trained model before to use this command.

    python train_tfrecord_dataset.py \
    	--dataset ./data/train_helmet_man_drill.tfrecord \
    	--val_dataset ./data/validation_helmet_man_drill.tfrecord \
    	--model yolov3 \
    	--classes ./data/OID/oidv4.names \
    	--num_classes 3 \
    	--mode fit --transfer darknet \
    	--batch_size 64 \
    	--epochs 10 \
    	--weights ./checkpoints/yolov3.tf \
    	--weights_num_classes 80 

if you have an OUT_OF_MEMORY error, reduce the batch_size or change the number of epochs. If you have NaN loss you can also increase the learning rate.

Inferences
==========

#### Image input folder 

Add your images in the Images/image_input and launch with your weights:

    python infer_images_folder.py \
    	--classes /data/oidv4.names \
    	--num_classes 3 \
    	--weights /checkpoints/yolov3_train_oidv4_1.tf \

You can see the result in the image_output folder.

#### by using a video

    python detect_video.py --weights ./checkpoints/yolov3-tiny.tf \
    --model yolov3-tiny \
    --video ./video.mp4 \
    --output ./video_tiny.avi

#### by using the webcam


    python detect_video.py --video 0 --classes ./data/coco.names \
    	--num_classes 80 \
    	--weights ./checkpoints/yolov3.tf

Future work
===========

* 

References
==========

https://github.com/zzh8829/yolov3-tf2
https://github.com/EscVM/OIDv4_ToolKit
https://github.com/zamblauskas/oidv4-toolkit-tfrecord-generator/

