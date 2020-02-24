# Pre-Forward Collision Warning Application (P-FCW)

Traffic crash in one the leading casue of death in United States (US). Every year alomost 30000 americans die in traffic related crahses.
According to the US Department of Transporation report, alomost 2685 rear end collision happened only in 2018. However, most of this rear end collision or forward collision can be prevented using latest technology. Many recent vehicles are equiped with advance driving assitance system (ADAS), where forward collision warning is included. However, I think, inspite of having a forward collision warning application, we can futher reduce the forward collision suing a pre-forward collision warning application. Hence, in this project I have implemented a pre-forward collision warning application using a vision-based system. Futhermore, one of my goal was to reduce the cost of such ADAS system, and equip the conventional vehciles with pre-forward collision applicaiton, whcih does not contain any ADAS system yet. 

In implemetnign the applicaion, I have integragted the lane detection and vehicle detection together. I have used the code from one of my project on Lane detection which I learned during "Self-driving Car Nano-degree" program, and used IntelOpenVino pre-trained model for vehicle detection. 


### Hardware Setup.


One of the focus of this project was to reduce the cost. Thus, I have a Raspberry Pi instead of using a GPU-enabled device (e.g. Jetson TX).
But the inference for vehicle detection was slow in Raspberry Pi, and bought an Intel Modivus Neural Computing Stick for inference engine.

Here is the list of hardware and their corresponding prices:


1. Raspberry Pi ($35)
2. Intel Neural Compute Stick ($79)
3. Video Web Camera ($70)

The hardware setup inside the vehicle is shown here:



### ALgorithm:
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
### Project Output
I installed the necessary hardware in the vehicle and drove my car in one of the road in Clemson (US-123). The output from the P-FCW application is shown in the following video:


### Performance:
