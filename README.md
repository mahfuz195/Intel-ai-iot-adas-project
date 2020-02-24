# Pre-Forward Collision Warning Application (P-FCW)

The roadway traffic crash in one the leading causes of death in United States (US). Every year almost 30000 Americans die in traffic related crashes. According to the US Department of Transportation report, almost 2685 rear-end collision happened only in 2018. However, most of this rear-end collision or forward-collision can be prevented using latest technology. Many recent vehicles are equipped with advance driving assistance system (ADAS), where forward collision warning is included. However, I think, in spite of having a forward collision warning, we can further reduce the forward collision using a pre-forward collision warning that will be given to the drivers. Hence, in this project I have implemented a pre-forward collision warning application using a computer vision. Furthermore, one of my goals was to reduce the cost of such ADAS system, and equip the conventional vehicles with pre-forward collision application, which does not contain any ADAS system yet.


In implementing the application, I have integrated the simple lane detection and Deep Learning based vehicle detection together. I have used the code from one of my projects for Lane detection, which I learned during "Self-driving Car Nano-degree" program, and used IntelOpenVino pre-trained model for vehicle detection.



### Hardware Setup


One of the focus of this project was to reduce the cost. Thus, I have a Raspberry Pi instead of using a GPU-enabled device (e.g. Jetson TX).
But the inference for vehicle detection was slow in Raspberry Pi, and bought an Intel Modivus Neural Computing Stick for inference engine.

Here is the list of hardware and their corresponding prices:


1. Raspberry Pi ($35)
2. Intel Neural Compute Stick ($79)
3. Video Web Camera ($70)

The hardware setup inside the vehicle is shown here:



### P-FCW ALgorithm
The algorithm is implemneted as follows:
```
while (capture image from dashboard):
  1. Detect Lane Marking.
  2. Form a polygon (P) using detected lane markings.
  3. Detect Vehicles
  4. For each vehicle detected:
     4(a) Calculate the bottom middle point (p1) of bounding box.
     4(b) if p1 is inside the P:
          --> issue and pre-forward collision warning.
```

### Project File Structure:

The project contains the following files: 

    .
    ├── app_pfcw.py            # Contains the implementaion of the P-FCW application.
    ├── helper.py               # Contains the helpers functions for Lane Detection.
    ├── imgs                    # Contains the raw images of the collected driving data.
    ├── models                  # Contains the pre-trained IntelOpenVino models.
    └── README.md
    


### Used Learning from IntelAI Course
In the foundational course of IntelAI using OpenVino, I have learned how to use a pre-trained model for inference. I have used this learning in vehicle detection. I have used the pre-train demo model : https://docs.openvinotoolkit.org/latest/_models_intel_vehicle_detection_adas_0002_description_vehicle_detection_adas_0002.html


Also, I have implemented  the Async Inference model for speeding up the inference time. The code I followed to implement is from Intel page:
https://github.com/opencv/open_model_zoo/tree/master/demos/python_demos


During the project, I learned a lot on IntelOpenVino tools and is happy to implement in real-world what I have learned.



### Project Output
I installed the necessary hardware in the vehicle and drove my car in one of the road in Clemson (US-123). The output from the P-FCW application is shown in the following video:


### How to run the project

To run the project, please go inside the project direction and run the following command in the terminal:

``` python3 app_pfcw.py```


### Further Improvement

The performace of the applicaiton in terms of frame per second (FPS) was a bottneck in my implementation. I was able to gain atmost ~5FPS for P-FCW usign Async mode of IntelOpenVino tools and Intel Neural Compute Stick. 

