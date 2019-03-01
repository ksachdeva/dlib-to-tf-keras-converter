# Dlib to tensorflow/keras converter

Dlib face recognition network is a nerual network trained by Davis King (https://github.com/davisking/dlib) using his C++ library/toolkit.

Please see more details about this network here at this link - https://github.com/davisking/dlib-models

## Objective

This repository provides set of scripts to convert dlib's face recognition network into other formats such as 

- Keras hd5
- Tensorflow's saved model and frozen graph
- ONNX [TODO]

## Usage and how it works

### Step 1

The weights for the model are in stored in the binary fomat (.dat file). You can download the file and unzip it.

Here is the location from you can download it - http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 

### Step 2

Dlib toolkit provides a method to take the serialized weights and generate the XML from it. This is done using a C++ tool `xml_generator`. All it really does is that it defines the network in C++ (following an example from dlib), loads
the weights and then serialize it.

In order to use this tool you should have following installed on your machine -

- dlib (and required dependencies)
- cmake

```bash
cd xml_generator
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

If you are able to compile this tool `xml_generator` then you would have `build/bin` directory in it which
contains the executable

```bash
# now simply do 
# [Make sure you have dlib_face_recognition_resnet_model_v1.dat in build directory]
./bin/xml_generator
```

You should now have dlib_face_recognition_resnet_model_v1.xml file in the build directory.

## Step 3

Time for the final convertion

```bash
# It's as simple as 
# [Make sure you are at the root of this repository]
python main.py --xml-weights xml_generator/build/dlib_face_recognition_resnet_model_v1.xml
```