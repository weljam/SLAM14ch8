#include <iostream>
#include <opencv2/core.hpp>
#include <mutex>
#include <sophus/se3.hpp>
#include <unistd.h>

using namespace std;
using namespace cv; 
mutex t_mutex;
void fun (const Range range)
{
  int cnt =0;
  for (size_t i = range.start; i < range.end; i++) {
    cnt++;
    for(int i = 0;i<100000;i++){
        for(int j = 0;j<100000;j++) double a = 1.0/2*3/2*3;
    }
    cout << "i: " << i << endl;
  }
  unique_lock<mutex> lck(t_mutex);
  cout<<"cnt:"<<cnt<<endl;
  //Sophus::SE3d T21;
  //cout<<"T21:"<<T21.matrix()<<endl;
}

int main ()
{
  parallel_for_(cv::Range(0, 100), &fun);
  cout<<"hello"<<endl;
  return 0;
}
