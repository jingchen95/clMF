#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
using namespace std;

int convertToString(const char *filename,string &s)
{
  size_t size;
  char* str;
  fstream f(filename,(fstream::in|fstream::binary));

  if(f.is_open())
  {
    size_t fileSize;
    f.seekg(0,fstream::end);
    size=fileSize=(size_t)f.tellg();
    f.seekg(0,fstream::beg);
    str=new char[size+1];
    if(!str)
    {
       f.close();
       return 0;
    }
    f.read(str,fileSize);
    f.close();
    str[size]='\0';
    s=str;
    delete[] str;
    return 0;
  }
  cout<<"Error:failed to open file:"<<filename<<"\n";
  return -1;
}

int getPlatform(cl_platform_id &platform)
{
  platform=NULL;
  cl_uint numPlatforms;
  cl_int status=clGetPlatformIDs(0,NULL,&numPlatforms);
  if(status!=CL_SUCCESS)
  {
    cout<<"ERROR:Getting platforms!\n";
    return -1;
  }
  if(numPlatforms>0)
  {
    cl_platform_id* platforms=(cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
    status=clGetPlatformIDs(numPlatforms,platforms,NULL);
    //platform=platforms[1];
    platform=platforms[0]; 
   free(platforms);
  }
  else
    return -1;
}

cl_device_id *getCl_device_id(cl_platform_id &platform)
{
  cl_uint numDevices=0;
  cl_device_id *devices=NULL;
  //cl_int status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_ACCELERATOR,0,NULL,&numDevices);
  //cl_int status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_CPU,0,NULL,&numDevices);
  cl_int status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,0,NULL,&numDevices);
  if(numDevices>0)
  {
    devices=(cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
    //status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_CPU,numDevices,devices,NULL);
    //status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_ACCELERATOR,numDevices,devices,NULL);
    status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,numDevices,devices,NULL);
  }
    return devices;
}
