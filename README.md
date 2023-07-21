# CudaProjects
A set of C++/CUDA projects making use of the GPU

## Building
If not already done, you need to [install CUDA](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) for your computer
### Windows
Use the visual studio solution file  
You can switch between command line mode and graphical mode by changing the Sub System option (Linker -> System) in project settings.  
Configurations ending with _RTX will produce the ray-tracing project, the others will produce the mandelbrot demo
### Linux
```make mandelbrot```  

To build the mandelbrot demo,  

```make ray-tracing```  

to build the ray-tracing project, and  


```make clean```  

to clean all of the build files

## Usage
Here are all of the options for command line mode :  

-> ```-r <resolution>```  

Sets the target resolution, <resolution> is either two positive number like this ```1920x1080```  
Or one of theses values : ```hd fhd 2k 4k 8k```  
Default is 1920x1080 (fhd)

-> ```-f <framerate>```  

Set the target framerate for the mandelbrot animation, <framerate> must be a positive number like this : ```30```  
Default is 30

-> ```-s <frame>```   

Sets the starting frame for the mandelbrot animation, <frame> must be a positive number like this : ```100```  
This is usefult if the program got interrupted and you don't want to start from the beginning  
Default is 0

-> ```-q <quality>```  

Set the quality for the ray-tracing program, <quality> must be a positive number like this : ```16```  
Default is 16
